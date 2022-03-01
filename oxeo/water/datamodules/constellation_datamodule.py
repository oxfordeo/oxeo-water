from typing import Callable, Dict, List, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from oxeo.core.logging import logger
from oxeo.satools.io import ConstellationData

from .datasets.virtual_dataset import VirtualDataset


def worker_init_fn(worker_id):
    """Configures each dataset worker process.

    Just has one job!  To call SatelliteDataset.per_worker_init().
    """
    # get_worker_info() returns information specific to each worker process.
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        logger.debug("worker_info is None!")
    else:
        dataset_obj = worker_info.dataset  # The Dataset copy in this worker process.
        dataset_obj.per_worker_init(worker_id=worker_info.id)


class ConstellationDataModule(LightningDataModule):
    def __init__(
        self,
        train_constellation_regions: Dict[str, List[List[ConstellationData]]],
        val_constellation_regions: Dict[str, List[List[ConstellationData]]] = None,
        patch_size: int = 100,
        train_index_map: Tuple[int, str, int, int] = None,
        val_index_map: Tuple[int, str, int, int] = None,
        preprocess: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        batch_size: int = 32,
        num_workers: int = 16,
    ):
        super().__init__()
        self.train_constellation_regions = train_constellation_regions
        self.val_constellation_regions = val_constellation_regions
        self.patch_size = patch_size
        self.train_index_map = train_index_map
        self.val_index_map = val_index_map

        self.preprocess = preprocess
        self.transforms = transforms
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        """This method is called N times (N being the number of GPUS)"""

        self.train_dataset = VirtualDataset(
            self.train_constellation_regions,
            self.patch_size,
            self.preprocess,
            self.transforms,
            index_map=self.train_index_map,
        )

        self.val_dataset = VirtualDataset(
            self.val_constellation_regions,
            self.patch_size,
            self.preprocess,
            None,
            index_map=self.val_index_map,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            shuffle=True,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            shuffle=False,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
        )
