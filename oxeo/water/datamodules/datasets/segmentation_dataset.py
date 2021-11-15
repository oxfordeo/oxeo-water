from typing import Callable, List, Optional

import numpy as np
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    """A dataset that reads from cache in local disk."""

    def __init__(
        self,
        patch_paths: List[str],
        transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.transform = transform
        self.patch_paths = patch_paths

    def __getitem__(self, index: int):
        patch_path = self.patch_paths[index]

        data = np.load(f"{patch_path}/data.npy").astype(np.int16)
        label = np.load(f"{patch_path}/weak_labels.npy")
        sample = {
            "data": data,
            "label": label,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.patch_paths)
