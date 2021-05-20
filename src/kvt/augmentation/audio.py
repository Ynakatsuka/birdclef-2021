try:
    import colorednoise as cn
except ImportError:
    cn = None
import cv2
import librosa
import numpy as np
from audiomentations.core.transforms_interface import BaseWaveformTransform
from scipy.signal import butter, lfilter


class PinkNoise(BaseWaveformTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5, max_snr=20):
        super().__init__(p)

        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, sr):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        pink_noise = cn.powerlaw_psd_gaussian(1, len(y))
        a_pink = np.sqrt(pink_noise ** 2).max()
        augmented = (y + pink_noise * 1 / a_pink * a_noise).astype(y.dtype)
        return augmented


def _db2float(db: float, amplitude=True):
    if amplitude:
        return 10 ** (db / 20)
    else:
        return 10 ** (db / 10)


def volume_down(y: np.ndarray, db: float):
    """
    Low level API for decreasing the volume
    Parameters
    ----------
    y: numpy.ndarray
        stereo / monaural input audio
    db: float
        how much decibel to decrease
    Returns
    -------
    applied: numpy.ndarray
        audio with decreased volume
    """
    applied = y * _db2float(-db)
    return applied


def volume_up(y: np.ndarray, db: float):
    """
    Low level API for increasing the volume
    Parameters
    ----------
    y: numpy.ndarray
        stereo / monaural input audio
    db: float
        how much decibel to increase
    Returns
    -------
    applied: numpy.ndarray
        audio with increased volume
    """
    applied = y * _db2float(db)
    return applied


class RandomVolume(BaseWaveformTransform):
    def __init__(self, always_apply=False, p=0.5, limit=10):
        super().__init__(p)
        self.limit = limit

    def apply(self, y: np.ndarray, sr):
        db = np.random.uniform(-self.limit, self.limit)
        if db >= 0:
            return volume_up(y, db)
        else:
            return volume_down(y, db)


class CosineVolume(BaseWaveformTransform):
    def __init__(self, always_apply=False, p=0.5, limit=10):
        super().__init__(p)
        self.limit = limit

    def apply(self, y: np.ndarray, sr):
        db = np.random.uniform(-self.limit, self.limit)
        cosine = np.cos(np.arange(len(y)) / len(y) * np.pi * 2)
        dbs = _db2float(cosine * db)
        return (y * dbs).astype("float32")


class TimeShift(BaseWaveformTransform):
    def __init__(self, always_apply=False, p=0.5):
        super().__init__(p)

    def apply(self, y: np.ndarray, sr):
        l = len(y) // 4
        start_ = int(np.random.uniform(-l, l))
        if start_ >= 0:
            y = np.r_[y[start_:], np.random.uniform(-0.001, 0.001, start_)]
        else:
            y = np.r_[np.random.uniform(-0.001, 0.001, -start_), y[:start_]]

        return y.astype("float32")


class SpeedTuning(BaseWaveformTransform):
    def __init__(self, always_apply=False, p=0.5, speed_range=0.3):
        super().__init__(p)
        self.speed_range = speed_range

    def apply(self, y: np.ndarray, sr):
        len_y = len(y)
        speed_rate = np.random.uniform(1 - self.speed_range, 1 + self.speed_range)
        y_speed_tune = cv2.resize(y, (1, int(len(y) * speed_rate))).squeeze()

        if len(y_speed_tune) < len_y:
            pad_len = len_y - len(y_speed_tune)
            y_speed_tune = np.r_[
                np.random.uniform(-0.001, 0.001, int(pad_len / 2)),
                y_speed_tune,
                np.random.uniform(-0.001, 0.001, int(np.ceil(pad_len / 2))),
            ]
        else:
            cut_len = len(y_speed_tune) - len_y
            y_speed_tune = y_speed_tune[int(cut_len / 2) : int(cut_len / 2) + len_y]

        return y_speed_tune.astype("float32")


class StretchAudio(BaseWaveformTransform):
    def __init__(self, always_apply=False, p=0.5):
        super().__init__(p)

    def apply(self, data, sr):
        input_length = len(data)

        data = librosa.effects.time_stretch(data, sr)

        if len(data) > input_length:
            data = data[:input_length]
        else:
            data = np.pad(data, (0, max(0, input_length - len(data))), "constant")

        return data.astype("float32")


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


class LowFrequencyMask(BaseWaveformTransform):
    """Ref: https://www.kaggle.com/vladimirsydor/4-th-place-solution-inference-and-training-tips?scriptVersionId=42796948"""

    def __init__(
        self,
        p: int = 0.5,
        allways_apply: bool = False,
        max_cutoff: float = 5,
        min_cutoff: float = 4,
    ):
        super().__init__(p)
        self.p = p
        self.allways_apply = allways_apply
        self.max_cutoff = max_cutoff
        self.min_cutoff = min_cutoff

    def apply(self, au, sr):
        if np.random.binomial(n=1, p=self.p) or self.allways_apply:
            cutoff_value = np.random.uniform(low=self.min_cutoff, high=self.max_cutoff)
            au = butter_lowpass_filter(au, cutoff=cutoff_value, fs=sr / 1000)

        return au


class OneOf:
    def __init__(self, transforms: list, p: float):
        self.transforms = transforms
        self.p = p

    def __call__(self, y: np.ndarray, sr):
        if np.random.rand() < self.p:
            n_trns = len(self.transforms)
            trns_idx = np.random.choice(n_trns)
            trns = self.transforms[trns_idx]
            y = trns(y, sr)
        return y
