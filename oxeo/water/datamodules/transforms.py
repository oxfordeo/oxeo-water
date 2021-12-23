from typing import List

import kornia
import torch
from torch import Tensor, nn


class MinMaxNormalize(nn.Module):
    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x) -> Tensor:
        x = kornia.enhance.normalize_min_max(x, min_val=0.0, max_val=1.0, eps=1e-06)
        return x


class MasksToLabel:
    """Merge given masks in order given order
    (mask 1 will be label 1, mask 2 will be label 2, etc)
    and create *label* key in sample"""

    def __init__(self, keys: List[str]):
        self.keys = keys

    def __call__(self, sample):
        label = sample[self.keys[0]]
        for i, key in enumerate(self.keys[1:]):
            label[sample[key] == 1] = i + 2
        sample["label"] = label.squeeze().long()
        return sample


class FilterZeros:
    """Filter the sample by converting it to None
    if zeros are above the threshold percentage"""

    def __init__(self, keys: List[str], percentage: float = 0.10):
        self.keys = keys
        self.percentage = percentage

    def __call__(self, sample):
        res = False
        for key in self.keys:
            arr = sample[key]
            non_zero_percentage = torch.count_nonzero(arr) / arr.nelement()
            res = res | (non_zero_percentage >= (1 - self.percentage))
            if res:
                break
        return sample if res else None


class SelectBands:
    """Computes sample virtual arrays to numpy array"""

    def __init__(self, bands):
        self.bands = bands

    def __call__(self, sample):
        sample["data"] = sample["data"].sel({"bands": self.bands})
        return sample


class SelectConstellation:
    """Computes sample virtual arrays to numpy array"""

    def __init__(self, constellation):
        self.constellation = constellation

    def __call__(self, sample):
        for key in sample.keys():
            sample[key] = sample[key][self.constellation]
        return sample


class Compute:
    """Computes sample virtual arrays to numpy array"""

    def __call__(self, sample):
        for key in sample.keys():
            sample[key] = sample[key].values.astype("<i2")
        return sample
