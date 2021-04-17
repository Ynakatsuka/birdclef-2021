import cv2
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform


class Line(ImageOnlyTransform):
    def __init__(self, magnitude=0.5, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.magnitude = magnitude

    def apply(self, image, **params):
        num_lines = int(round(1 + np.random.randint(8) * self.magnitude))
        height, width = image.shape[:2]
        image = image.copy()

        def line0():
            return (0, 0), (width - 1, 0)

        def line1():
            return (0, height - 1), (width - 1, height - 1)

        def line2():
            return (0, 0), (0, height - 1)

        def line3():
            return (width - 1, 0), (width - 1, height - 1)

        def line4():
            x0, x1 = np.random.choice(width, 2)
            return (x0, 0), (x1, height - 1)

        def line5():
            y0, y1 = np.random.choice(height, 2)
            return (0, y0), (width - 1, y1)

        for i in range(num_lines):
            p = np.array([1 / 4, 1 / 4, 1 / 4, 1 / 4, 1, 1])
            func = np.random.choice(
                [line0, line1, line2, line3, line4, line5], p=p / p.sum()
            )
            (x0, y0), (x1, y1) = func()

            colorr = np.random.randint(0, 50)
            colorg = np.random.randint(0, 50)
            colorb = np.random.randint(0, 50)
            thickness = np.random.randint(1, 5)
            line_type = np.random.choice([cv2.LINE_AA, cv2.LINE_4, cv2.LINE_8])

            cv2.line(
                image,
                (x0, y0),
                (x1, y1),
                (colorr, colorg, colorb),
                thickness,
                line_type,
            )
        return image
