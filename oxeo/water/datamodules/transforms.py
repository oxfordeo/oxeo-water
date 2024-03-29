from typing import Dict, List

import kornia
import numpy as np
import torch
import torchvision.transforms.functional as F
from torch import Tensor, nn


class MinMaxNormalize(nn.Module):
    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x) -> Tensor:
        x = kornia.enhance.normalize_min_max(x, min_val=0.0, max_val=1.0, eps=1e-06)
        return x


class ConstellationNormalize:
    def __init__(
        self,
        band_mean: Dict[str, Dict[str, List]] = None,
        band_std: Dict[str, Dict[str, List]] = None,
        bands: List[str] = None,
    ):
        self.band_mean = band_mean
        self.band_std = band_std
        self.bands = bands

    def __call__(self, sample):
        if sample is None:
            return sample
        constellation_mean = self.band_mean[sample["constellation"]]
        constellation_mean = [constellation_mean[x] for x in constellation_mean.keys()]
        constellation_std = self.band_std[sample["constellation"]]
        constellation_std = [constellation_std[x] for x in constellation_std.keys()]
        img = sample["image"]
        img = F.normalize(img.float(), constellation_mean, constellation_std)
        sample["image"] = img
        return sample


class ZimmozToLabel:
    def __call__(self, sample):
        if sample is None:
            return sample
        label = sample["zimmoz"]
        label[label > 2] = 0
        sample["label"] = label.squeeze().long()
        return sample


class MasksToLabel:
    """Merge given masks in order given order
    (mask 1 will be label 1, mask 2 will be label 2, etc)
    and create *label* key in sample"""

    def __init__(self, keys: List[str]):
        self.keys = keys

    def __call__(self, sample):
        if sample is None:
            return sample
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
            arr = sample[key].numpy()
            non_zero_percentage = np.count_nonzero(arr) / arr.size
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
