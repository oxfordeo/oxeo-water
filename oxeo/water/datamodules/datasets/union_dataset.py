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
        ds1_tile_dates = self.datasets[0].tile_dates
        ds2_tile_dates = self.datasets[1].tile_dates

        self.tile_dates = {}
        for d in [ds1_tile_dates, ds2_tile_dates]:
            for key in d:
                try:
                    self.tile_dates[key] = np.union1d(self.tile_dates[key], d[key])
                except KeyError:
                    self.tile_dates[key] = d[key]

    def __getitem__(self, index):
        tile_id, timestamp, _, _, _ = index

        # Not all datasets are guaranteed to have a valid query
        samples = []
        for ds in self.datasets:
            if ds.valid_date(tile_id, timestamp):
                samples.append(ds[index])

        return self.collate_fn(samples)

    def __len__(self):
        return len(self.tile_dates)

    def valid_date(self, tile_id: str, timestamp):
        dates = self.tile_dates.get(tile_id)
        if dates is not None:
            return timestamp in dates
        else:
            return False

    def per_worker_init(self) -> None:
        for ds in self.datasets:
            ds.per_worker_init()
