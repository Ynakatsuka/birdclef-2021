from albumentations.core.transforms_interface import ImageOnlyTransform
import cv2
import numpy as np


class Sprinkle(ImageOnlyTransform):

    def __init__(self, size=16, magnitude=0.5, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.size = size
        self.magnitude = magnitude

    def apply(self, image, **params):
        num_sprinkle = int(round(1 + np.random.randint(10) * self.magnitude))
        image = image.copy()
        image_small = cv2.resize(image, dsize=None, fx=0.25, fy=0.25)
        m = np.where(image_small > 255 * 0.25)
        num = len(m[0])
        if num == 0: return image

        s = self.size // 2
        i = np.random.choice(num, num_sprinkle)
        for y, x in zip(m[0][i], m[1][i]):
            y = y * 4 + 2
            x = x * 4 + 2
            image[y - s:y + s, x - s:x + s] = 0  # 0.5 #1 #
        return image
