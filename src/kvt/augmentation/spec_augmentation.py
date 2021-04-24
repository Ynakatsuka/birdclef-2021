# Ref: https://github.com/qiuqiangkong/torchlibrosa/blob/master/torchlibrosa/augmentation.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DropStripes(nn.Module):
    def __init__(self, dim, drop_width, stripes_num):
        """Drop stripes.

        Args:
          dim: int, dimension along which to drop
          drop_width: int, maximum width of stripes to drop
          stripes_num: int, how many stripes to drop
        """
        super(DropStripes, self).__init__()

        assert dim in [2, 3]  # dim 2: time; dim 3: frequency

        self.dim = dim
        self.drop_width = drop_width
        self.stripes_num = stripes_num

    def forward(self, input):
        """input: (batch_size, channels, time_steps, freq_bins)"""

        assert input.ndimension() == 4

        if self.training is False:
            return input

        else:
            batch_size = input.shape[0]
            total_width = input.shape[self.dim]

            shuffle_batch = input[
                torch.randint(low=0, high=batch_size, size=(batch_size,))
            ]

            for n in range(batch_size):
                self.transform_slice(input[n], total_width, shuffle_batch[n])

            return input

    def transform_slice(self, e, total_width, s):
        """e, s: (channels, time_steps, freq_bins)"""

        for _ in range(self.stripes_num):
            distance = torch.randint(low=0, high=self.drop_width, size=(1,))[0]
            bgn = torch.randint(low=0, high=total_width - distance, size=(1,))[0]

            if self.dim == 2:
                e[:, bgn : bgn + distance, :] = 0
            elif self.dim == 3:
                e[:, :, bgn : bgn + distance] = 0


class MixStripes(DropStripes):
    def transform_slice(self, e, total_width, s):
        """e, s: (channels, time_steps, freq_bins)"""

        for _ in range(self.stripes_num):
            distance = torch.randint(low=0, high=self.drop_width, size=(1,))[0]
            bgn = torch.randint(low=0, high=total_width - distance, size=(1,))[0]

            if self.dim == 2:
                e[:, bgn : bgn + distance, :] = (
                    e[:, bgn : bgn + distance, :] + s[:, bgn : bgn + distance, :]
                ) / 2
            elif self.dim == 3:
                e[:, :, bgn : bgn + distance] = (
                    e[:, :, bgn : bgn + distance] + s[:, :, bgn : bgn + distance]
                ) / 2


class CutStripes(DropStripes):
    def transform_slice(self, e, total_width, s):
        """e, s: (channels, time_steps, freq_bins)"""

        for _ in range(self.stripes_num):
            distance = torch.randint(low=0, high=self.drop_width, size=(1,))[0]
            bgn = torch.randint(low=0, high=total_width - distance, size=(1,))[0]

            if self.dim == 2:
                e[:, bgn : bgn + distance, :] = s[:, bgn : bgn + distance, :]
            elif self.dim == 3:
                e[:, :, bgn : bgn + distance] = s[:, :, bgn : bgn + distance]


class SpecAugmentationPlusPlus(nn.Module):
    def __init__(
        self,
        time_drop_width,
        time_stripes_num,
        freq_drop_width,
        freq_stripes_num,
        method="zm",
    ):
        """SpecAugment++.
        https://arxiv.org/pdf/2103.16858.pdf

        Args:
          time_drop_width: int
          time_stripes_num: int
          freq_drop_width: int
          freq_stripes_num: int
        """

        super(SpecAugmentationPlusPlus, self).__init__()
        assert method in ("zm", "mm", "cm")
        if method == "zm":
            dropper = DropStripes
        elif method == "mm":
            dropper = MixStripes
        elif method == "cm":
            dropper = CutStripes

        self.time_dropper = dropper(
            dim=2, drop_width=time_drop_width, stripes_num=time_stripes_num
        )
        self.freq_dropper = dropper(
            dim=3, drop_width=freq_drop_width, stripes_num=freq_stripes_num
        )

    def forward(self, input):
        x = self.time_dropper(input)
        x = self.freq_dropper(x)
        return x


if __name__ == "__main__":

    torch.manual_seed(0)
    random_state = np.random.RandomState(0)
    np_data = random_state.normal(size=(10, 4, 640, 64))
    pt_data = torch.Tensor(np_data)

    spec_augmenter = SpecAugmentationPlusPlus(
        time_drop_width=64, time_stripes_num=2, freq_drop_width=16, freq_stripes_num=2
    )

    # Training stage
    spec_augmenter.train()  # set to spec_augmenter.eval() for evaluation
    result = spec_augmenter(pt_data)

    print(result.shape)
