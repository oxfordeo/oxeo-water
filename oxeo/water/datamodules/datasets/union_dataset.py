from typing import Any, Callable, Dict, Sequence

import numpy as np
from torch.utils.data import Dataset

from .utils import merge_samples


class UnionDataset(Dataset):
    """A dataset that reads from zarr tiles."""

    def __init__(
        self,
        dataset1: Dataset,
        dataset2: Dataset,
        collate_fn: Callable[
            [Sequence[Dict[str, Any]]], Dict[str, Any]
        ] = merge_samples,
    ):
        """Pytorch Dataset to load data from tiles paths


        Args:
            tile_paths (List[TilePath]): a list of Tile paths to load data from
            transform (Optional[Callable], optional): Transformations to apply to each sample. Defaults to None.
        """
        super().__init__()
        self.datasets = [dataset1, dataset2]
        self.collate_fn = collate_fn

        # Merge dataset dates
        self._merge_dataset_dates()

    def _merge_dataset_dates(self):
        ds1_tile_dates = self.datasets[0].dates
        ds2_tile_dates = self.datasets[1].dates
        self.dates = []
        for i in range(len(ds1_tile_dates)):
            self.dates.append(np.union1d(ds1_tile_dates[i], ds2_tile_dates[i]))

    def __getitem__(self, index):
        tile_index, timestamp, _, _, _ = index

        # Not all datasets are guaranteed to have a valid query
        samples = []
        for ds in self.datasets:
            if ds.valid_date(tile_index, timestamp):
                samples.append(ds[index])

        return self.collate_fn(samples)

    def __len__(self):
        return len(self.datasets[0])

    def valid_date(self, tile_index: int, timestamp):
        return timestamp in self.dates[tile_index]

    def per_worker_init(self) -> None:
        for ds in self.datasets:
            ds.per_worker_init()
