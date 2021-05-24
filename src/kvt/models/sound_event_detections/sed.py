import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kvt.augmentation import SpecAugmentationPlusPlus, mixup
from kvt.models.layers import Flatten, GeM
from torchlibrosa.augmentation import SpecAugmentation
from torchlibrosa.stft import LogmelFilterBank, Spectrogram

from .audio_features import (
    Loudness,
    PCENTransform,
    add_frequency_encoding,
    add_time_encoding,
    make_delta,
)


def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


def init_bn(bn):
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


def init_weights(model):
    classname = model.__class__.__name__
    if classname.find("Conv2d") != -1:
        nn.init.xavier_uniform_(model.weight, gain=np.sqrt(2))
        model.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)
    elif classname.find("GRU") != -1:
        for weight in model.parameters():
            if len(weight.size()) > 1:
                nn.init.orghogonal_(weight.data)
    elif classname.find("Linear") != -1:
        model.weight.data.normal_(0, 0.01)
        model.bias.data.zero_()


def interpolate(x: torch.Tensor, ratio: int):
    """Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output: torch.Tensor, frames_num: int):
    """Pad framewise_output to the same length as input frames. The pad value
    is the same as the value of the last frame.
    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad
    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    output = F.interpolate(
        framewise_output.unsqueeze(1),
        size=(frames_num, framewise_output.size(2)),
        align_corners=True,
        mode="bilinear",
    ).squeeze(1)

    return output


class AttBlockV2(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation="linear"):
        super().__init__()

        self.activation = activation
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)

    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == "linear":
            return x
        elif self.activation == "sigmoid":
            return torch.sigmoid(x)


def gem(x, kernel_size, p=3, eps=1e-6):
    return F.avg_pool1d(x.clamp(min=eps).pow(p), kernel_size).pow(1.0 / p)


def do_mixup(x, lam, indices):
    shuffled_x = x[indices]
    x = lam * x + (1 - lam) * shuffled_x
    return x


class SED(nn.Module):
    def __init__(
        self,
        encoder,
        in_features,
        num_classes,
        n_fft,
        hop_length,
        sample_rate,
        n_mels,
        fmin,
        fmax,
        dropout_rate=0.5,
        freeze_spectrogram_parameters=True,
        freeze_logmel_parameters=True,
        use_spec_augmentation=True,
        time_drop_width=64,
        time_stripes_num=2,
        freq_drop_width=8,
        freq_stripes_num=2,
        spec_augmentation_method=None,
        apply_mixup=False,
        apply_spec_shuffle=False,
        spec_shuffle_prob=0,
        use_gru_layer=False,
        apply_tta=False,
        use_loudness=False,
        use_spectral_centroid=False,
        apply_delta_spectrum=False,
        apply_time_freq_encoding=False,
        min_db=120,
        apply_pcen=False,
        freeze_pcen_parameters=False,
        use_multisample_dropout=False,
        multisample_dropout=0.5,
        num_multisample_dropout=5,
        pooling_kernel_size=3,
        **params,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.dropout_rate = dropout_rate
        self.apply_mixup = apply_mixup
        self.apply_spec_shuffle = apply_spec_shuffle
        self.spec_shuffle_prob = spec_shuffle_prob
        self.use_gru_layer = use_gru_layer
        self.apply_tta = apply_tta
        self.use_loudness = use_loudness
        self.use_spectral_centroid = use_spectral_centroid
        self.apply_delta_spectrum = apply_delta_spectrum
        self.apply_time_freq_encoding = apply_time_freq_encoding
        self.apply_pcen = apply_pcen
        self.use_multisample_dropout = use_multisample_dropout
        self.num_multisample_dropout = num_multisample_dropout
        self.pooling_kernel_size = pooling_kernel_size

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window="hann",
            center=True,
            pad_mode="reflect",
            freeze_parameters=freeze_spectrogram_parameters,
        )

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            ref=1.0,
            amin=1e-10,
            top_db=None,
            freeze_parameters=freeze_logmel_parameters,
        )

        # Spec augmenter
        self.spec_augmenter = None
        if use_spec_augmentation and (spec_augmentation_method is None):
            self.spec_augmenter = SpecAugmentation(
                time_drop_width=time_drop_width,
                time_stripes_num=time_stripes_num,
                freq_drop_width=freq_drop_width,
                freq_stripes_num=freq_stripes_num,
            )
        elif use_spec_augmentation and (spec_augmentation_method is not None):
            self.spec_augmenter = SpecAugmentationPlusPlus(
                time_drop_width=time_drop_width,
                time_stripes_num=time_stripes_num,
                freq_drop_width=freq_drop_width,
                freq_stripes_num=freq_stripes_num,
                method=spec_augmentation_method,
            )

        if self.use_gru_layer:
            self.gru = nn.GRU(in_features, in_features, batch_first=True)

        if self.use_loudness:
            self.loudness_bn = nn.BatchNorm1d(1)
            self.loudness_extractor = Loudness(
                sr=sample_rate,
                n_fft=n_fft,
                min_db=min_db,
            )

        if self.use_spectral_centroid:
            self.spectral_centroid_bn = nn.BatchNorm1d(1)

        if self.apply_pcen:
            self.pcen_transform = PCENTransform(
                trainable=~freeze_pcen_parameters,
            )

        self.bn0 = nn.BatchNorm2d(n_mels)

        # layers = list(encoder.children())[:-2]
        # self.encoder = nn.Sequential(*layers)
        self.encoder = encoder

        if self.use_multisample_dropout:
            self.big_dropout = nn.Dropout(p=multisample_dropout)

        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        self.att_block = AttBlockV2(in_features, num_classes, activation="sigmoid")

        self.init_weight()

    def init_weight(self):
        init_layer(self.fc1)
        init_bn(self.bn0)

    def forward(self, input, mixup_lambda=None, mixup_index=None):
        # (batch_size, 1, time_steps, freq_bins)
        x = self.spectrogram_extractor(input)

        additional_features = []
        if self.use_loudness:
            loudness = self.loudness_extractor(x)
            loudness = self.loudness_bn(loudness)
            loudness = loudness.unsqueeze(-1)
            loudness = loudness.repeat(1, 1, 1, self.n_mels)
            additional_features.append(loudness)

        if self.use_spectral_centroid:
            spectral_centroid = x.mean(-1)
            spectral_centroid = self.spectral_centroid_bn(spectral_centroid)
            spectral_centroid = spectral_centroid.unsqueeze(-1)
            spectral_centroid = spectral_centroid.repeat(1, 1, 1, self.n_mels)
            additional_features.append(spectral_centroid)

        # logmel
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)
        frames_num = x.shape[2]
        x = x.transpose(1, 3).contiguous()
        x = self.bn0(x)
        x = x.transpose(1, 3).contiguous()

        if (
            self.training
            and self.apply_spec_shuffle
            and (np.random.rand() < self.spec_shuffle_prob)
        ):
            # (batch_size, 1, time_steps, freq_bins)
            idx = torch.randperm(x.shape[3])
            x = x[:, :, :, idx]

        if (self.training or self.apply_tta) and (self.spec_augmenter is not None):
            x = self.spec_augmenter(x)

        # additional features
        if self.apply_delta_spectrum:
            delta_1 = make_delta(x)
            delta_2 = make_delta(delta_1)
            additional_features.extend([delta_1, delta_2])

        if self.apply_time_freq_encoding:
            freq_encode = add_frequency_encoding(x)
            time_encode = add_time_encoding(x)
            additional_features.extend([freq_encode, time_encode])

        if self.apply_pcen:
            pcen = self.pcen_transform(x)
            additional_features.append(pcen)

        if len(additional_features) > 0:
            additional_features.append(x)
            x = torch.cat(additional_features, dim=1)

        # Mixup on spectrogram
        if self.training and self.apply_mixup and (mixup_lambda is not None):
            x = do_mixup(x, mixup_lambda, mixup_index)

        x = x.transpose(2, 3).contiguous()
        # (batch_size, channels, freq, frames)
        x = self.encoder(x)

        # (batch_size, channels, frames)
        x = torch.mean(x, dim=2)

        # GRU
        if self.use_gru_layer:
            # (batch_size, channels, frames) -> (batch_size, channels, frames)
            x, _ = self.gru(x.transpose(1, 2).contiguous())
            x = x.transpose(1, 2).contiguous()

        # channel smoothing
        # (batch_size, channels, frames)
        x = gem(x, kernel_size=self.pooling_kernel_size)

        if self.use_multisample_dropout:
            x = x.transpose(1, 2).contiguous()
            x = torch.mean(
                torch.stack(
                    [
                        F.relu_(self.fc1(self.big_dropout(x)))
                        for _ in range(self.num_multisample_dropout)
                    ],
                    dim=0,
                ),
                dim=0,
            )
            x = x.transpose(1, 2).contiguous()
        else:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x = x.transpose(1, 2).contiguous()

            x = F.relu_(self.fc1(x))
            x = x.transpose(1, 2).contiguous()

        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        logit = torch.sum(norm_att * self.att_block.cla(x), dim=2)
        segmentwise_logit = self.att_block.cla(x).transpose(1, 2).contiguous()
        segmentwise_output = segmentwise_output.transpose(1, 2).contiguous()

        interpolate_ratio = frames_num // segmentwise_output.size(1)

        # Get framewise output
        framewise_output = interpolate(segmentwise_output, interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)

        framewise_logit = interpolate(segmentwise_logit, interpolate_ratio)
        framewise_logit = pad_framewise_output(framewise_logit, frames_num)

        output_dict = {
            "framewise_output": framewise_output,
            "segmentwise_output": segmentwise_output,
            "logit": logit,
            "framewise_logit": framewise_logit,
            "clipwise_output": clipwise_output,
        }

        return output_dict


class SEDV2(SED):
    """additional inputs"""

    def __init__(
        self,
        encoder,
        in_features,
        num_classes,
        n_fft,
        hop_length,
        sample_rate,
        n_mels,
        fmin,
        fmax,
        **params,
    ):
        super().__init__(
            encoder,
            in_features + 2,
            num_classes,
            n_fft,
            hop_length,
            sample_rate,
            n_mels,
            fmin,
            fmax,
            **params,
        )

        self.fc2 = nn.Linear(2, 2, bias=True)

    def forward(self, input, x_additional, mixup_lambda=None, mixup_index=None):
        # (batch_size, 1, time_steps, freq_bins)
        x = self.spectrogram_extractor(input)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        frames_num = x.shape[2]

        x = x.transpose(1, 3).contiguous()
        x = self.bn0(x)
        x = x.transpose(1, 3).contiguous()

        if (
            self.training
            and self.apply_spec_shuffle
            and (np.random.rand() < self.spec_shuffle_prob)
        ):
            # (batch_size, 1, time_steps, freq_bins)
            idx = torch.randperm(x.shape[3])
            x = x[:, :, :, idx]

        if (self.training or self.apply_tta) and (self.spec_augmenter is not None):
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and self.apply_mixup and (mixup_lambda is not None):
            x = do_mixup(x, mixup_lambda, mixup_index)

        x = x.transpose(2, 3).contiguous()
        # (batch_size, channels, freq, frames)
        x = self.encoder(x)

        # (batch_size, channels, frames)
        x = torch.mean(x, dim=2)

        # channel smoothing
        # (batch_size, channels, frames)
        x = gem(x, kernel_size=3)

        # (batch_size, channels+2, frames)
        y = self.fc2(x_additional)
        x = torch.cat(
            (x, y.unsqueeze(-1).repeat((1, 1, x.shape[-1]))),
            dim=1,
        )

        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = x.transpose(1, 2).contiguous()

        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2).contiguous()

        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        logit = torch.sum(norm_att * self.att_block.cla(x), dim=2)
        segmentwise_logit = self.att_block.cla(x).transpose(1, 2).contiguous()
        segmentwise_output = segmentwise_output.transpose(1, 2).contiguous()

        interpolate_ratio = frames_num // segmentwise_output.size(1)

        # Get framewise output
        framewise_output = interpolate(segmentwise_output, interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)

        framewise_logit = interpolate(segmentwise_logit, interpolate_ratio)
        framewise_logit = pad_framewise_output(framewise_logit, frames_num)

        output_dict = {
            "framewise_output": framewise_output,
            "segmentwise_output": segmentwise_output,
            "logit": logit,
            "framewise_logit": framewise_logit,
            "clipwise_output": clipwise_output,
        }

        return output_dict


class SEDV3(SED):
    """densenet"""

    def __init__(
        self,
        encoder,
        in_features,
        num_classes,
        n_fft,
        hop_length,
        sample_rate,
        n_mels,
        fmin,
        fmax,
        **params,
    ):
        super().__init__(
            encoder,
            in_features,
            num_classes,
            n_fft,
            hop_length,
            sample_rate,
            n_mels,
            fmin,
            fmax,
            **params,
        )

        self.encoder = encoder.features


class SEDV4(SED):
    """call/song prediction"""

    def __init__(
        self,
        encoder,
        in_features,
        num_classes,
        n_fft,
        hop_length,
        sample_rate,
        n_mels,
        fmin,
        fmax,
        **params,
    ):
        super().__init__(
            encoder,
            in_features,
            num_classes,
            n_fft,
            hop_length,
            sample_rate,
            n_mels,
            fmin,
            fmax,
            **params,
        )

        self.fc_type = nn.Linear(num_classes, 2, bias=True)

    def forward(self, input, mixup_lambda=None, mixup_index=None):
        # (batch_size, 1, time_steps, freq_bins)
        x = self.spectrogram_extractor(input)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        frames_num = x.shape[2]

        x = x.transpose(1, 3).contiguous()
        x = self.bn0(x)
        x = x.transpose(1, 3).contiguous()

        if (
            self.training
            and self.apply_spec_shuffle
            and (np.random.rand() < self.spec_shuffle_prob)
        ):
            # (batch_size, 1, time_steps, freq_bins)
            idx = torch.randperm(x.shape[3])
            x = x[:, :, :, idx]

        if (self.training or self.apply_tta) and (self.spec_augmenter is not None):
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and self.apply_mixup and (mixup_lambda is not None):
            x = do_mixup(x, mixup_lambda, mixup_index)

        x = x.transpose(2, 3).contiguous()
        # (batch_size, channels, freq, frames)
        x = self.encoder(x)

        # (batch_size, channels, frames)
        x = torch.mean(x, dim=2)

        # channel smoothing
        # (batch_size, channels, frames)
        x = gem(x, kernel_size=3)

        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = x.transpose(1, 2).contiguous()

        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2).contiguous()

        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)

        logit = torch.sum(norm_att * self.att_block.cla(x), dim=2)
        type_logit = self.fc_type(clipwise_output)

        segmentwise_logit = self.att_block.cla(x).transpose(1, 2).contiguous()
        segmentwise_output = segmentwise_output.transpose(1, 2).contiguous()

        interpolate_ratio = frames_num // segmentwise_output.size(1)

        # Get framewise output
        framewise_output = interpolate(segmentwise_output, interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)

        framewise_logit = interpolate(segmentwise_logit, interpolate_ratio)
        framewise_logit = pad_framewise_output(framewise_logit, frames_num)

        output_dict = {
            "framewise_output": framewise_output,
            "segmentwise_output": segmentwise_output,
            "logit": logit,
            "type_logit": type_logit,
            "framewise_logit": framewise_logit,
            "clipwise_output": clipwise_output,
        }

        return output_dict


class ImageSED(nn.Module):
    def __init__(
        self,
        encoder,
        in_features,
        num_classes,
        n_fft,
        hop_length,
        sample_rate,
        n_mels,
        fmin,
        fmax,
        dropout_rate=0.5,
        freeze_spectrogram_parameters=True,
        freeze_logmel_parameters=True,
        use_spec_augmentation=True,
        time_drop_width=64,
        time_stripes_num=2,
        freq_drop_width=8,
        freq_stripes_num=2,
        spec_augmentation_method=None,
        apply_mixup=False,
        apply_spec_shuffle=False,
        spec_shuffle_prob=0,
        use_gru_layer=False,
        apply_tta=False,
        use_loudness=False,
        use_spectral_centroid=False,
        apply_delta_spectrum=False,
        apply_time_freq_encoding=False,
        min_db=120,
        apply_pcen=False,
        freeze_pcen_parameters=False,
        use_multisample_dropout=False,
        multisample_dropout=0.5,
        num_multisample_dropout=5,
        pooling_kernel_size=3,
        **params,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.dropout_rate = dropout_rate
        self.apply_mixup = apply_mixup
        self.apply_spec_shuffle = apply_spec_shuffle
        self.spec_shuffle_prob = spec_shuffle_prob
        self.use_gru_layer = use_gru_layer
        self.apply_tta = apply_tta
        self.use_loudness = use_loudness
        self.use_spectral_centroid = use_spectral_centroid
        self.apply_delta_spectrum = apply_delta_spectrum
        self.apply_time_freq_encoding = apply_time_freq_encoding
        self.apply_pcen = apply_pcen
        self.use_multisample_dropout = use_multisample_dropout
        self.num_multisample_dropout = num_multisample_dropout
        self.pooling_kernel_size = pooling_kernel_size

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window="hann",
            center=True,
            pad_mode="reflect",
            freeze_parameters=freeze_spectrogram_parameters,
        )

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            ref=1.0,
            amin=1e-10,
            top_db=None,
            freeze_parameters=freeze_logmel_parameters,
        )

        # Spec augmenter
        self.spec_augmenter = None
        if use_spec_augmentation and (spec_augmentation_method is None):
            self.spec_augmenter = SpecAugmentation(
                time_drop_width=time_drop_width,
                time_stripes_num=time_stripes_num,
                freq_drop_width=freq_drop_width,
                freq_stripes_num=freq_stripes_num,
            )
        elif use_spec_augmentation and (spec_augmentation_method is not None):
            self.spec_augmenter = SpecAugmentationPlusPlus(
                time_drop_width=time_drop_width,
                time_stripes_num=time_stripes_num,
                freq_drop_width=freq_drop_width,
                freq_stripes_num=freq_stripes_num,
                method=spec_augmentation_method,
            )

        if self.use_gru_layer:
            self.gru = nn.GRU(in_features, in_features, batch_first=True)

        if self.use_loudness:
            self.loudness_bn = nn.BatchNorm1d(1)
            self.loudness_extractor = Loudness(
                sr=sample_rate,
                n_fft=n_fft,
                min_db=min_db,
            )

        if self.use_spectral_centroid:
            self.spectral_centroid_bn = nn.BatchNorm1d(1)

        if self.apply_pcen:
            self.pcen_transform = PCENTransform(
                trainable=~freeze_pcen_parameters,
            )

        self.encoder = encoder

    def forward(self, input, mixup_lambda=None, mixup_index=None):
        # (batch_size, 1, time_steps, freq_bins)
        x = self.spectrogram_extractor(input)

        additional_features = []
        if self.use_loudness:
            loudness = self.loudness_extractor(x)
            loudness = self.loudness_bn(loudness)
            loudness = loudness.unsqueeze(-1)
            loudness = loudness.repeat(1, 1, 1, self.n_mels)
            additional_features.append(loudness)

        if self.use_spectral_centroid:
            spectral_centroid = x.mean(-1)
            spectral_centroid = self.spectral_centroid_bn(spectral_centroid)
            spectral_centroid = spectral_centroid.unsqueeze(-1)
            spectral_centroid = spectral_centroid.repeat(1, 1, 1, self.n_mels)
            additional_features.append(spectral_centroid)

        # logmel
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        if (
            self.training
            and self.apply_spec_shuffle
            and (np.random.rand() < self.spec_shuffle_prob)
        ):
            # (batch_size, 1, time_steps, freq_bins)
            idx = torch.randperm(x.shape[3])
            x = x[:, :, :, idx]

        if (self.training or self.apply_tta) and (self.spec_augmenter is not None):
            x = self.spec_augmenter(x)

        # additional features
        if self.apply_delta_spectrum:
            delta_1 = make_delta(x)
            delta_2 = make_delta(delta_1)
            additional_features.extend([delta_1, delta_2])

        if self.apply_time_freq_encoding:
            freq_encode = add_frequency_encoding(x)
            time_encode = add_time_encoding(x)
            additional_features.extend([freq_encode, time_encode])

        if self.apply_pcen:
            pcen = self.pcen_transform(x)
            additional_features.append(pcen)

        if len(additional_features) > 0:
            additional_features.append(x)
            x = torch.cat(additional_features, dim=1)

        # Mixup on spectrogram
        if self.training and self.apply_mixup and (mixup_lambda is not None):
            x = do_mixup(x, mixup_lambda, mixup_index)

        x = x.transpose(2, 3).contiguous()
        # (batch_size, channels, freq, frames)
        x = self.encoder(x)

        return x
