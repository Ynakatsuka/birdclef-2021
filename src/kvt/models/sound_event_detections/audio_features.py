import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_deltas(
    specgram: torch.Tensor, win_length: int = 5, mode: str = "replicate"
) -> torch.Tensor:
    r"""Compute delta coefficients of a tensor, usually a spectrogram:

    .. math::
       d_t = \frac{\sum_{n=1}^{\text{N}} n (c_{t+n} - c_{t-n})}{2 \sum_{n=1}^{\text{N}} n^2}

    where :math:`d_t` is the deltas at time :math:`t`,
    :math:`c_t` is the spectrogram coeffcients at time :math:`t`,
    :math:`N` is ``(win_length-1)//2``.

    Args:
        specgram (Tensor): Tensor of audio of dimension (..., freq, time)
        win_length (int, optional): The window length used for computing delta (Default: ``5``)
        mode (str, optional): Mode parameter passed to padding (Default: ``"replicate"``)

    Returns:
        Tensor: Tensor of deltas of dimension (..., freq, time)

    Example
        >>> specgram = torch.randn(1, 40, 1000)
        >>> delta = compute_deltas(specgram)
        >>> delta2 = compute_deltas(delta)
    """
    device = specgram.device
    dtype = specgram.dtype

    # pack batch
    shape = specgram.size()
    specgram = specgram.reshape(1, -1, shape[-1])

    assert win_length >= 3

    n = (win_length - 1) // 2

    # twice sum of integer squared
    denom = n * (n + 1) * (2 * n + 1) / 3

    specgram = torch.nn.functional.pad(specgram, (n, n), mode=mode)

    kernel = torch.arange(-n, n + 1, 1, device=device, dtype=dtype).repeat(
        specgram.shape[1], 1, 1
    )

    output = (
        torch.nn.functional.conv1d(specgram, kernel, groups=specgram.shape[1]) / denom
    )

    # unpack batch
    output = output.reshape(shape)

    return output


def make_delta(input_tensor: torch.Tensor):
    input_tensor = input_tensor.transpose(3, 2)
    input_tensor = compute_deltas(input_tensor)
    input_tensor = input_tensor.transpose(3, 2)
    return input_tensor


class Loudness(nn.Module):
    def __init__(self, sr, n_fft, min_db):
        super().__init__()
        self.min_db = min_db
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        self.a_weighting = torch.nn.Parameter(
            data=torch.from_numpy(librosa.A_weighting(freqs + 1e-10)),
            requires_grad=False,
        )

    def forward(self, spec):
        power_db = torch.log10(spec ** 0.5 + 1e-10)

        loudness = power_db + self.a_weighting

        # loudness -= 10 * torch.log10(spec)
        loudness -= 20.7

        loudness = torch.clamp(loudness, min=-self.min_db)

        # Average over frequency bins.
        loudness = torch.mean(loudness, axis=-1).float()
        return loudness


def add_frequency_encoding(x):
    n, d, h, w = x.size()

    vertical = torch.linspace(-1, 1, w, device=x.device).view(1, 1, 1, -1)
    vertical = vertical.repeat(n, 1, h, 1)

    return vertical


def add_time_encoding(x):
    n, d, h, w = x.size()

    horizontal = torch.linspace(-1, 1, h, device=x.device).view(1, 1, -1, 1)
    horizontal = horizontal.repeat(n, 1, 1, w)

    return horizontal


class F2M(nn.Module):
    """This turns a normal STFT into a MEL Frequency STFT, using a conversion
       matrix.  This uses triangular filter banks.
    Args:
        n_mels (int): number of MEL bins
        sr (int): sample rate of audio signal
        f_max (float, optional): maximum frequency. default: sr // 2
        f_min (float): minimum frequency. default: 0
    """

    def __init__(
        self, n_mels=40, sr=16000, f_max=None, f_min=0.0, n_fft=40, onesided=True
    ):
        super().__init__()
        self.n_mels = n_mels
        self.sr = sr
        self.f_max = f_max if f_max is not None else sr // 2
        self.f_min = f_min
        self.n_fft = n_fft
        if onesided:
            self.n_fft = self.n_fft // 2 + 1
        self._init_buffers()

    def _init_buffers(self):
        m_min = 0.0 if self.f_min == 0 else 2595 * np.log10(1.0 + (self.f_min / 700))
        m_max = 2595 * np.log10(1.0 + (self.f_max / 700))

        m_pts = torch.linspace(m_min, m_max, self.n_mels + 2)
        f_pts = 700 * (10 ** (m_pts / 2595) - 1)

        bins = torch.floor(((self.n_fft - 1) * 2) * f_pts / self.sr).long()

        fb = torch.zeros(self.n_fft, self.n_mels)
        for m in range(1, self.n_mels + 1):
            f_m_minus = bins[m - 1].item()
            f_m = bins[m].item()
            f_m_plus = bins[m + 1].item()

            if f_m_minus != f_m:
                fb[f_m_minus:f_m, m - 1] = (
                    torch.arange(f_m_minus, f_m) - f_m_minus
                ) / (f_m - f_m_minus)
            if f_m != f_m_plus:
                fb[f_m:f_m_plus, m - 1] = (f_m_plus - torch.arange(f_m, f_m_plus)) / (
                    f_m_plus - f_m
                )
        self.register_buffer("fb", fb)

    def forward(self, spec_f):
        spec_m = torch.matmul(
            spec_f, self.fb
        )  # (c, l, n_fft) dot (n_fft, n_mels) -> (c, l, n_mels)
        return spec_m


def pcen(
    x,
    eps=1e-6,
    s=0.025,
    alpha=0.98,
    delta=2,
    r=0.5,
    training=False,
    last_state=None,
    empty=True,
):
    frames = x.split(1, -2)
    m_frames = []
    if empty:
        last_state = None
    for frame in frames:
        if last_state is None:
            last_state = frame
            m_frames.append(frame)
            continue
        if training:
            m_frame = ((1 - s) * last_state).add_(s * frame)
        else:
            m_frame = (1 - s) * last_state + s * frame
        last_state = m_frame
        m_frames.append(m_frame)
    M = torch.cat(m_frames, 1)
    if training:
        pcen_ = (x / (M + eps).pow(alpha) + delta).pow(r) - delta ** r
    else:
        pcen_ = x.div_(M.add_(eps).pow_(alpha)).add_(delta).pow_(r).sub_(delta ** r)
    return pcen_


class PCENTransform(nn.Module):
    """Ref: https://www.kaggle.com/simongrest/trainable-pcen-frontend-in-pytorch"""

    def __init__(self, eps=1e-6, s=0.025, alpha=0.98, delta=2, r=0.5, trainable=True):
        super().__init__()
        if trainable:
            self.log_s = nn.Parameter(torch.log(torch.Tensor([s])))
            self.log_alpha = nn.Parameter(torch.log(torch.Tensor([alpha])))
            self.log_delta = nn.Parameter(torch.log(torch.Tensor([delta])))
            self.log_r = nn.Parameter(torch.log(torch.Tensor([r])))
        else:
            self.s = s
            self.alpha = alpha
            self.delta = delta
            self.r = r
        self.eps = eps
        self.trainable = trainable

    def forward(self, x):
        x = x.permute((0, 1, 3, 2)).squeeze(dim=1)
        if self.trainable:
            x = pcen(
                x,
                self.eps,
                torch.exp(self.log_s),
                torch.exp(self.log_alpha),
                torch.exp(self.log_delta),
                torch.exp(self.log_r),
                self.training and self.trainable,
            )
        else:
            x = pcen(
                x,
                self.eps,
                self.s,
                self.alpha,
                self.delta,
                self.r,
                self.training and self.trainable,
            )
        x = x.unsqueeze(dim=1).permute((0, 1, 3, 2))
        return x
