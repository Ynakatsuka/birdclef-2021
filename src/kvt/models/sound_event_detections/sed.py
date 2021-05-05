import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kvt.augmentation import SpecAugmentationPlusPlus, mixup
from torchlibrosa.augmentation import SpecAugmentation
from torchlibrosa.stft import LogmelFilterBank, Spectrogram


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
        **params,
    ):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.apply_mixup = apply_mixup
        self.apply_spec_shuffle = apply_spec_shuffle
        self.spec_shuffle_prob = spec_shuffle_prob

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

        self.bn0 = nn.BatchNorm2d(n_mels)

        # layers = list(encoder.children())[:-2]
        # self.encoder = nn.Sequential(*layers)
        self.encoder = encoder

        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        self.att_block = AttBlockV2(in_features, num_classes, activation="sigmoid")

        self.init_weight()

    def init_weight(self):
        init_layer(self.fc1)
        init_bn(self.bn0)

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

        if self.training and (self.spec_augmenter is not None):
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
            dropout_rate,
            freeze_spectrogram_parameters,
            freeze_logmel_parameters,
            use_spec_augmentation,
            time_drop_width,
            time_stripes_num,
            freq_drop_width,
            freq_stripes_num,
            spec_augmentation_method,
            apply_mixup,
            apply_spec_shuffle,
            spec_shuffle_prob,
        )

        self.fc2 = nn.Linear(2, 2, bias=True)

    def forward(self, input, additional_x, mixup_lambda=None, mixup_index=None):
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

        if self.training and (self.spec_augmenter is not None):
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
        y = self.fc2(additional_x)
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
            dropout_rate,
            freeze_spectrogram_parameters,
            freeze_logmel_parameters,
            use_spec_augmentation,
            time_drop_width,
            time_stripes_num,
            freq_drop_width,
            freq_stripes_num,
            spec_augmentation_method,
            apply_mixup,
            apply_spec_shuffle,
            spec_shuffle_prob,
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
            dropout_rate,
            freeze_spectrogram_parameters,
            freeze_logmel_parameters,
            use_spec_augmentation,
            time_drop_width,
            time_stripes_num,
            freq_drop_width,
            freq_stripes_num,
            spec_augmentation_method,
            apply_mixup,
            apply_spec_shuffle,
            spec_shuffle_prob,
        )

        self.fc_type = nn.Linear(2, 2, bias=True)

    def forward(self, input, additional_x, mixup_lambda=None, mixup_index=None):
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

        if self.training and (self.spec_augmenter is not None):
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
        y = self.fc2(additional_x)
        x = torch.cat(
            (x, y.unsqueeze(-1).repeat((1, 1, x.shape[-1]))),
            dim=1,
        )

        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = x.transpose(1, 2).contiguous()

        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2).contiguous()

        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)

        type_logit = self.fc_type(norm_att * self.att_block.cla(x))

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
            "type_logit": type_logit,
            "framewise_logit": framewise_logit,
            "clipwise_output": clipwise_output,
        }

        return output_dict
