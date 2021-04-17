import cv2
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform


class RandomErasing(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.5, sl=0.02, sh=0.4, r1=0.3, r2=3.3):
        super().__init__(always_apply, p)
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.r2 = r2

    def apply(self, img, **params):
        target_img = img.copy()

        H, W, _ = target_img.shape
        S = H * W

        while True:
            Se = np.random.uniform(self.sl, self.sh) * S  # 画像に重畳する矩形の面積
            re = np.random.uniform(self.r1, self.r2)  # 画像に重畳する矩形のアスペクト比

            He = int(np.sqrt(Se * re))  # 画像に重畳する矩形のHeight
            We = int(np.sqrt(Se / re))  # 画像に重畳する矩形のWidth

            xe = np.random.randint(0, W)  # 画像に重畳する矩形のx座標
            ye = np.random.randint(0, H)  # 画像に重畳する矩形のy座標

            if xe + We <= W and ye + He <= H:
                # 画像に重畳する矩形が画像からはみ出していなければbreak
                break

        mask = np.random.randint(0, 255, (He, We, 3))  # 矩形がを生成 矩形内の値はランダム値
        target_img[ye : ye + He, xe : xe + We, :] = mask  # 画像に矩形を重畳

        return target_img
