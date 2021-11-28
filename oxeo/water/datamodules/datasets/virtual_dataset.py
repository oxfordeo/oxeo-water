from datetime import datetime
from typing import Callable, List, Optional

import xarray as xr
from torch.utils.data import Dataset


class VirtualDataset(Dataset):
    """A dataset that reads from cache in local disk."""

    def __init__(
        self,
        data_virtual_datasets: List[xr.Dataset],
        label_virtual_datasets: List[xr.Dataset] = None,
        patch_size: int = 100,
        start_date: datetime = None,
        end_date: datetime = None,
        transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.data_virtual_datasets = [
            ds.sel({"revisits": slice(start_date, end_date)})
            for ds in data_virtual_datasets
        ]
        self.label_virtual_datasets = label_virtual_datasets
        if self.label_virtual_datasets is not None:
            self.label_virtual_datasets = [
                ds.sel({"revisits": slice(start_date, end_date)})
                for ds in label_virtual_datasets
            ]

            self.data_virtual_datasets = [
                ds.sel({"revisits": label_virtual_datasets[i]["revisits"]})
                for i, ds in enumerate(data_virtual_datasets)
            ]

        self.patch_size = patch_size
        self.start_date = start_date
        self.end_date = end_date
        self.transform = transform
        self.index_map = []
        for dataset_index in range(len(data_virtual_datasets)):
            timestamps = self.data_virtual_datasets[dataset_index]["revisits"].data
            for timestamp_index in timestamps:
                for i in range(
                    self.data_virtual_datasets[dataset_index].dims["height"]
                    // self.patch_size
                ):
                    for j in range(
                        self.data_virtual_datasets[dataset_index].dims["width"]
                        // self.patch_size
                    ):
                        self.index_map.append((dataset_index, timestamp_index, i, j))

    def __getitem__(self, index: int):

        dataset_index, timestamp_index, i, j = self.index_map[index]

        data = (
            self.data_virtual_datasets[dataset_index]
            .sel({"revisits": timestamp_index})
            .isel(
                {
                    "height": slice(
                        i * self.patch_size, i * self.patch_size + self.patch_size
                    ),
                    "width": slice(
                        j * self.patch_size, j * self.patch_size + self.patch_size
                    ),
                }
            )
        )

        sample = {
            "data": data,
        }

        if self.label_virtual_datasets is not None:

            label = (
                self.label_virtual_datasets[dataset_index]
                .sel({"revisits": timestamp_index})
                .isel(
                    {
                        "height": slice(
                            i * self.patch_size,
                            i * self.patch_size + self.patch_size,
                        ),
                        "width": slice(
                            j * self.patch_size, j * self.patch_size + self.patch_size
                        ),
                    }
                )
            )
            sample["label"] = label

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.index_map)
