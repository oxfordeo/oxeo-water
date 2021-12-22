from typing import Callable, Dict, List, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from oxeo.water.datamodules.datasets import IterableTileDataset
from oxeo.water.models.utils import TilePath, tile_from_id

from .utils import notnone_collate_fn


def worker_init_fn(worker_id):
    """Configures each dataset worker process.

    Just has one job!  To call SatelliteDataset.per_worker_init().
    """
    # get_worker_info() returns information specific to each worker process.
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        print("worker_info is None!")
    else:
        dataset_obj = worker_info.dataset  # The Dataset copy in this worker process.
        dataset_obj.per_worker_init()


class TileDataModule(LightningDataModule):
    def __init__(
        self,
        train_constellation_tile_ids: Dict[str, List[str]],
        val_constellation_tile_ids: Dict[str, List[str]] = None,
        bands: Tuple[str, ...] = ("nir", "red", "green", "blue", "swir1", "swir2"),
        masks: Tuple[str, ...] = ("pekel", "cloud_mask"),
        target_size: int = 1000,
        chip_size: int = 256,
        revisits_per_epoch: int = 500,
        samples_per_revisit: int = 10020,
        transforms: Optional[Callable] = None,
        batch_size: int = 32,
        num_workers: int = 1,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.transforms = transforms
        self.train_constellation_tile_paths = [
            TilePath(tile_from_id(tile_id), k)
            for k, v in train_constellation_tile_ids.items()
            for tile_id in v
        ]
        self.val_constellation_tile_paths = [
            TilePath(tile_from_id(tile_id), k)
            for k, v in val_constellation_tile_ids.items()
            for tile_id in v
        ]
        self.bands = bands
        self.masks = tuple(masks)
        self.target_size = target_size
        self.chip_size = chip_size
        self.revisits_per_epoch = revisits_per_epoch
        self.samples_per_revisit = samples_per_revisit
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def create_dataset(
        self, constellation_tile_paths: List[List[TilePath]], sampler: str = "random"
    ):

        ds = IterableTileDataset(
            constellation_tile_paths,
            transform=self.transforms,
            masks=self.masks,
            target_size=self.target_size,
            chip_size=self.chip_size,
            bands=self.bands,
            revisits_per_epoch=self.revisits_per_epoch,
            samples_per_revisit=self.samples_per_revisit,
            sampler=sampler,
        )

        return ds

    def setup(self, stage=None):
        """This method is called N times (N being the number of GPUS)"""
        self.train_dataset = self.create_dataset(
            self.train_constellation_tile_paths, sampler="random"
        )
        self.val_dataset = self.create_dataset(
            self.val_constellation_tile_paths, sampler="grid"
        )
        if self.num_workers == 0:
            self.train_dataset.per_worker_init()
            self.val_dataset.per_worker_init()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=notnone_collate_fn,
            drop_last=True,
            pin_memory=self.pin_memory,
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=self.pin_memory,
            worker_init_fn=worker_init_fn,
        )
