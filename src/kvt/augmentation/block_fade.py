from albumentations.core.transforms_interface import ImageOnlyTransform
import cv2
import numpy as np


class BlockFade(ImageOnlyTransform):
    def __init__(self, magnitude=0.5, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.magnitude = magnitude

    def apply(self, image, **params):
        size = [0.1, self.magnitude]

        height, width = image.shape[:2]

        # get bounding box
        m = image.copy()
        cv2.rectangle(m, (0, 0), (height, width), 1, 5)
        m = image < 0.5
        if m.sum() == 0: return image

        m = np.where(m)
        y0, y1, x0, x1 = np.min(m[0]), np.max(m[0]), np.min(m[1]), np.max(m[1])
        w = x1 - x0
        h = y1 - y0
        if w * h < 10: return image

        ew, eh = np.random.uniform(*size, 2)
        ew = int(ew * w)
        eh = int(eh * h)

        ex = np.random.randint(0, w - ew) + x0
        ey = np.random.randint(0, h - eh) + y0

        image[ey:ey + eh, ex:ex + ew] *= np.random.randint(0.1, 0.5)  # 1 #
        image = np.clip(image, 0, 255)
        image = image.astype(np.int)
        return image
