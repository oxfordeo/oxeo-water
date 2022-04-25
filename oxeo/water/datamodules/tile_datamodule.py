from typing import Dict, List, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from oxeo.core.logging import logger
from oxeo.core.models.tile import TilePath, tile_from_id
from oxeo.water.datamodules.constants import (
    CONSTELLATION_BAND_MEAN,
    CONSTELLATION_BAND_STD,
)
from oxeo.water.datamodules.datasets import TileDataset
from oxeo.water.datamodules.samplers import RandomSampler

from .transforms import ConstellationNormalize, FilterZeros, MasksToLabel, ZimmozToLabel
from .utils import notnone_collate_fn


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
        batch_size: int = 32,
        num_workers: int = 1,
        pin_memory: bool = False,
        cache_dir: str = None,
        cache_bytes: int = None,
        train_start_date: str = "0001-01-01",
        train_end_date: str = "9999-01-01",
        val_start_date: str = "0001-01-01",
        val_end_date: str = "9999-01-01",
        root_dir: str = "gs://oxeo-water/prod",
        valid_dates: List[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        if "zimmoz" in masks:
            to_label_tf = ZimmozToLabel()
        else:
            to_label_tf = MasksToLabel(masks)
        self.transforms = Compose(
            [
                to_label_tf,
                FilterZeros(keys=["label"], percentage=0.99),
                ConstellationNormalize(
                    CONSTELLATION_BAND_MEAN, CONSTELLATION_BAND_STD, bands
                ),
            ]
        )
        self.train_constellation_tile_paths = [
            TilePath(tile_from_id(tile_id), k, root_dir)
            for k, v in train_constellation_tile_ids.items()
            for tile_id in v
        ]
        self.val_constellation_tile_paths = [
            TilePath(tile_from_id(tile_id), k, root_dir)
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
        self.cache_dir = cache_dir
        self.cache_bytes = cache_bytes
        self.train_start_date = train_start_date
        self.train_end_date = train_end_date
        self.val_start_date = val_start_date
        self.val_end_date = val_end_date
        self.valid_dates = valid_dates

    def create_dataset(
        self,
        constellation_tile_paths: List[List[TilePath]],
        start_date: str,
        end_date: str,
        valid_dates: Dict[str, Dict[str, List[str]]],
    ):
        ds = TileDataset(
            constellation_tile_paths,
            transform=self.transforms,
            masks=self.masks,
            target_size=self.target_size,
            bands=self.bands,
            cache_dir=self.cache_dir,
            cache_bytes=self.cache_bytes,
            start_date=start_date,
            end_date=end_date,
            valid_dates=valid_dates,
        )

        return ds

    def setup(self, stage=None):
        """This method is called N times (N being the number of GPUS)"""
        self.train_dataset = self.create_dataset(
            self.train_constellation_tile_paths,
            self.train_start_date,
            self.train_end_date,
            self.valid_dates,
        )
        self.val_dataset = self.create_dataset(
            self.val_constellation_tile_paths,
            self.val_start_date,
            self.val_end_date,
            self.valid_dates,
        )
        if self.num_workers == 0:
            self.train_dataset.per_worker_init()
            self.val_dataset.per_worker_init()

    def train_dataloader(self):

        sampler = RandomSampler(
            self.train_dataset,
            chip_size=self.chip_size,
            revisits_per_epoch=self.revisits_per_epoch,
            samples_per_revisit=self.samples_per_revisit,
        )

        return DataLoader(
            self.train_dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=notnone_collate_fn,
            drop_last=True,
            pin_memory=self.pin_memory,
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self):

        sampler = RandomSampler(
            self.val_dataset,
            chip_size=self.chip_size,
            revisits_per_epoch=self.revisits_per_epoch,
            samples_per_revisit=self.samples_per_revisit,
        )

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=notnone_collate_fn,
            drop_last=True,
            pin_memory=self.pin_memory,
            worker_init_fn=worker_init_fn,
        )
