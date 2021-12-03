import math
import random
from concurrent import futures
from typing import Callable, Dict, List, Optional, Tuple

import gcsfs
import torch
from fsspec import asyn
from loguru import logger
from satools.io import ConstellationData, load_virtual_datasets
from torch.utils.data import IterableDataset


class VirtualDataset(IterableDataset):
    """A dataset that reads from xarray datasets."""

    def __init__(
        self,
        constellation_regions: Dict[str, List[List[ConstellationData]]],
        patch_size: int = 100,
        preprocess: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        index_map: List[Tuple[int, str, int, int]] = None,
        shuffle_index: bool = False,
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

        """
        super().__init__()

        self.constellation_regions = constellation_regions
        self.patch_size = patch_size
        self.preprocess = preprocess
        self.transform = transform

        self.index_map = index_map
        logger.info(f"There are {len(self.index_map)} instances.")
        if shuffle_index:
            random.shuffle(self.index_map)

        self.dates = [item[1] for item in self.index_map]
        self.date_range = (min(self.dates), max(self.dates))

    def __iter__(self):
        self.index_iter = iter(self.index_map)

        with futures.ThreadPoolExecutor(max_workers=1) as executor:
            future_data = executor.submit(self.load_data_to_memory)
            for _ in range(len(self.index_map)):
                sample = future_data.result()
                future_data = executor.submit(self.load_data_to_memory)

                if self.preprocess:
                    sample = self.preprocess(sample)

                if self.transform:
                    sample = self.transform(sample)

                yield sample

    def load_data_to_memory(self):
        dataset_index, timestamp_index, i, j = next(self.index_iter)

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
            "data": data["sentinel-2"].values.astype("<i2"),
        }

        if self.label_virtual_datasets is not None:

            label = self.get_dataset_patch(
                self.label_virtual_datasets[dataset_index], timestamp_index, i, j
            )
            sample["label"] = label["sentinel-2"].values.astype("<i2")

        return sample

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

        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset

        overall_start = 0
        overall_end = len(self.index_map)
        # configure the dataset to only process the split workload
        per_worker = int(
            math.ceil((overall_end - overall_start) / float(worker_info.num_workers))
        )
        worker_id = worker_info.id
        start = overall_start + worker_id * per_worker
        end = min(start + per_worker, overall_end)

        dataset.index_map = dataset.index_map[start:end]

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
