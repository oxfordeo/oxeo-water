from typing import Callable, Dict, List, Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from oxeo.water.datamodules.datasets import TileDataset, UnionDataset
from oxeo.water.datamodules.samplers import GridSampler, RandomSampler
from oxeo.water.models.utils import TilePath, tile_from_id


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
        bands: List[str] = None,
        tile_size: int = 1000,
        chip_size: int = 256,
        transforms: Optional[Callable] = None,
        total_samples: int = 2000,
        batch_size: int = 32,
        num_workers: int = 1,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.transforms = transforms
        self.train_constellation_tile_paths = [
            [TilePath(tile_from_id(tile_id), k) for tile_id in v]
            for k, v in train_constellation_tile_ids.items()
        ]
        self.val_constellation_tile_paths = [
            [TilePath(tile_from_id(tile_id), k) for tile_id in v]
            for k, v in val_constellation_tile_ids.items()
        ]
        self.bands = bands
        self.tile_size = tile_size
        self.chip_size = chip_size
        self.total_samples = total_samples
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def create_dataset(self, constellation_tile_paths: List[List[TilePath]]):
        first_constellation_tiles = constellation_tile_paths[0]

        ds = TileDataset(
            first_constellation_tiles,
            transform=self.transforms,
            masks=("pekel",),
            target_size=self.tile_size,
            bands=self.bands,
        )
        for tile_paths in constellation_tile_paths[1:]:
            ds2 = TileDataset(
                tile_paths,
                transform=self.transforms,
                masks=("pekel",),
                target_size=self.tile_size,
                bands=self.bands,
            )

            ds = UnionDataset(ds, ds2)
        return ds

    def setup(self, stage=None):
        """This method is called N times (N being the number of GPUS)"""
        self.train_dataset = self.create_dataset(self.train_constellation_tile_paths)
        self.val_dataset = self.create_dataset(self.val_constellation_tile_paths)
        if self.num_workers == 0:
            self.train_dataset.per_worker_init()
            self.val_dataset.per_worker_init()

    def train_dataloader(self):
        sampler = RandomSampler(
            self.train_dataset, self.tile_size, self.chip_size, self.total_samples
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=self.pin_memory,
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self):
        sampler = GridSampler(self.val_dataset, self.tile_size, self.chip_size)

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=self.pin_memory,
            worker_init_fn=worker_init_fn,
        )
