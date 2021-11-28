from typing import Any

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from .datasets.virtual_dataset import VirtualDataset


class SegmentationDataModule(LightningDataModule):
    def __init__(
        self,
        transforms: Any = None,
        batch_size: int = 32,
        num_workers: int = 16,
        train_paths: str = None,
        val_paths: str = None,
    ):
        super().__init__()
        self.transforms = transforms
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_paths = train_paths  # to be populated in prepare
        self.val_paths = val_paths

    def setup(self, stage=None):
        """This method is called N times (N being the number of GPUS)

        Args:
            stage ([type], optional): pl stage. Defaults to None.
        """

        self.train_dataset = VirtualDataset(
            patch_paths=self.train_paths,
            transform=self.transforms,
        )

        self.val_dataset = VirtualDataset(
            patch_paths=self.val_paths,
            transform=self.transforms,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            shuffle=False,
            pin_memory=True,
        )
