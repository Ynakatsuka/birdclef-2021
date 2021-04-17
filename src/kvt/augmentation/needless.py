import os
import random

import cv2
from albumentations.core.transforms_interface import ImageOnlyTransform


class NeedleAugmentation(ImageOnlyTransform):
    def __init__(
        self,
        always_apply=False,
        p=0.5,
        n_needles=2,
        dark_needles=False,
        needle_folder="data/input/external_data/xray-needle-augmentation",
    ):
        super().__init__(always_apply, p)
        self.n_needles = n_needles
        self.dark_needles = dark_needles
        self.needle_folder = needle_folder

    def apply(self, image, **params):
        height, width, _ = image.shape  # target image width and height
        needle_images = [im for im in os.listdir(self.needle_folder) if "png" in im]

        for _ in range(1, self.n_needles):
            needle = cv2.cvtColor(
                cv2.imread(
                    os.path.join(self.needle_folder, random.choice(needle_images))
                ),
                cv2.COLOR_BGR2RGB,
            )
            needle = cv2.flip(needle, random.choice([-1, 0, 1]))
            needle = cv2.rotate(needle, random.choice([0, 1, 2]))

            h_height, h_width, _ = needle.shape  # needle image width and height
            roi_ho = random.randint(0, abs(image.shape[0] - needle.shape[0]))
            roi_wo = random.randint(0, abs(image.shape[1] - needle.shape[1]))
            roi = image[roi_ho : roi_ho + h_height, roi_wo : roi_wo + h_width]

            # Creating a mask and inverse mask
            img2gray = cv2.cvtColor(needle, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            # Now black-out the area of needle in ROI
            img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

            # Take only region of insect from insect image.
            if self.dark_needles:
                img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
                needle_fg = cv2.bitwise_and(img_bg, img_bg, mask=mask)
            else:
                needle_fg = cv2.bitwise_and(needle, needle, mask=mask)

            # Put needle in ROI and modify the target image
            dst = cv2.add(img_bg, needle_fg, dtype=cv2.CV_64F)

            image[roi_ho : roi_ho + h_height, roi_wo : roi_wo + h_width] = dst

        return image
