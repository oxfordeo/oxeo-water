from typing import Callable, Iterable, List, Optional

import zarr
from torch.utils.data import Dataset

from oxeo.water.models.utils import TilePath, load_tile


class TileDataset(Dataset):
    """A dataset that reads from zarr tiles."""

    def __init__(
        self,
        tile_paths: List[TilePath],
        transform: Optional[Callable] = None,
        masks: Iterable[str] = None,
    ):
        """Pytorch Dataset to load data from tiles paths


        Args:
            tile_paths (List[TilePath]): a list of Tile paths to load data from
            transform (Optional[Callable], optional): Transformations to apply to each sample. Defaults to None.
        """
        super().__init__()

        self.tile_paths = tile_paths
        self.transform = transform
        self.masks = masks

        self.dates = [
            zarr.open_array(tile_path.timestamps_path).shape[0]
            for tile_path in self.tile_paths
        ]

    def __getitem__(self, index: int):
        tile_index, timestamp_index, i, j, chip_size = index

        sample = load_tile(self.tile_paths[tile_index], masks=tuple(self.masks))
        for key in sample.keys():
            sample[key] = sample[key][
                timestamp_index, ..., i : i + chip_size, j : j + chip_size
            ]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.tile_paths)
