"""
Ref: https://github.com/facebookresearch/CovidPrognosis/blob/1c8625d36ef7ff1b19c457cd6b6e8543d4635c21/covidprognosis/data/transforms.py
"""
import cv2
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform


class HistogramNormalize(ImageOnlyTransform):
    """
    Apply histogram normalization.
    Args:
        number_bins: Number of bins to use in histogram.
    """

    def __init__(self, number_bins=256, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.number_bins = number_bins

    def apply(self, image, **params):
        # get image histogram
        image_histogram, bins = np.histogram(
            image.flatten(), self.number_bins, density=True
        )
        cdf = image_histogram.cumsum()  # cumulative distribution function
        cdf = 255 * cdf / cdf[-1]  # normalize

        # use linear interpolation of cdf to find new pixel values
        image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

        return image_equalized.reshape(image.shape).astype("float32")
