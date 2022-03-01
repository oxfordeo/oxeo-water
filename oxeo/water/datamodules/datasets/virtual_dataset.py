from typing import Callable, Dict, List, Optional, Tuple

import gcsfs
from fsspec import asyn
from torch.utils.data import Dataset

from oxeo.core.logging import logger
from oxeo.satools.io import ConstellationData, load_virtual_datasets


class VirtualDataset(Dataset):
    """A dataset that reads from xarray datasets."""

    def __init__(
        self,
        constellation_regions: Dict[str, List[List[ConstellationData]]],
        patch_size: int = 100,
        preprocess: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        index_map: List[Tuple[int, str, int, int]] = None,
        label_delta: int = None,
    ):
        """Pytorch Dataset to load from virtual xarray datasets

        Args:
            constellation_regions (Dict[str, List[List[ConstellationData]]]): a dictionary that must contain a 'data' key and
                                an optional 'label' key. The values should be a list of lists of ConstellationData that
                                will be used to build xarrays.
            preprocess (Optional[Callable], optional): Preprocess to apply to the resulting sample. Defaults to None.
            transform (Optional[Callable], optional): Transforms to apply to the resulting sample. Defaults to None.
            index_map: List[Tuple[int, str, int, int]]: Index to each patch in the datasets. Use create_index_map function
                                to create it. It must be done outside of the VirtualDataset because otherwise
                                there are tilt problems with ffspec in multiprocessing.
            label_delta: int : If set takes actual label - delta range of labels, and puts uncertainty values
                                in max(label_deltas) - labels

        """
        super().__init__()

        self.constellation_regions = constellation_regions
        self.patch_size = patch_size
        self.preprocess = preprocess
        self.transform = transform
        self.index_map = index_map
        self.label_delta = label_delta

        self.dates = [item[1] for item in self.index_map]
        self.date_range = (min(self.dates), max(self.dates))

    def __getitem__(self, index: int):
        dataset_index, timestamp_index, i, j = self.index_map[index]

        # This is only if using this class without datamodule
        if not hasattr(self, "data_virtual_datasets"):
            (
                self.data_virtual_datasets,
                self.label_virtual_datasets,
            ) = load_virtual_datasets(self.constellation_regions, self.date_range, None)

        data = self.get_dataset_patch(
            self.data_virtual_datasets[dataset_index], timestamp_index, i, j
        )
        sample = {
            "data": data,
        }

        if self.label_virtual_datasets is not None:

            label = self.get_dataset_patch(
                self.label_virtual_datasets[dataset_index], timestamp_index, i, j
            )

            if self.label_delta:
                actual_timestamp_index = self.dates.index(timestamp_index)
                if actual_timestamp_index > self.label_delta:
                    past_timestamp = self.dates[
                        actual_timestamp_index - self.label_delta
                    ]

                    label = self.get_dataset_patch(
                        self.label_virtual_datasets[dataset_index],
                        slice(past_timestamp, timestamp_index),
                        i,
                        j,
                    )

                    logger.debug(label)

            sample["label"] = label

        if self.preprocess:
            sample = self.preprocess(sample)

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.index_map)

    def per_worker_init(self, worker_id: int = 0) -> None:
        """
        This is needed to work with gcp and pytorch datamodules.
        Otherwise it won't work when num_workers > 0
        """
        asyn.iothread[0] = None
        asyn.loop[0] = None

        gcs = gcsfs.GCSFileSystem(
            access="read_only",
            # skip_instance_cache=True  # Why skip_instance_cache?  See https://github.com/dask/gcsfs/issues/379#issuecomment-839929801
        )
        fs_mapper = gcs.get_mapper

        """Called by worker_init_fn on each copy of VirtualDataset after the worker process has been spawned."""
        (
            self.data_virtual_datasets,
            self.label_virtual_datasets,
        ) = load_virtual_datasets(
            self.constellation_regions, self.date_range, fs_mapper
        )

    def get_dataset_patch(self, ds, timestamp_index, i, j):
        return ds.sel({"revisits": timestamp_index}).isel(
            {
                "height": slice(
                    i * self.patch_size, i * self.patch_size + self.patch_size
                ),
                "width": slice(
                    j * self.patch_size, j * self.patch_size + self.patch_size
                ),
            }
        )
