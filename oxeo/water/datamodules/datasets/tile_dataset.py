from typing import Callable, Iterable, List, Optional

import gcsfs
import zarr
from fsspec import asyn
from torch.utils.data import Dataset

from oxeo.water.models.utils import TilePath, load_tile


class TileDataset(Dataset):
    """A dataset that reads from zarr tiles."""

    def __init__(
        self,
        tile_paths: List[TilePath],
        transform: Optional[Callable] = None,
        masks: Iterable[str] = (),
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
        self.fs_mapper = None

        self.dates = [
            zarr.open_array(tile_path.timestamps_path).shape[0]
            for tile_path in self.tile_paths
        ]

    def __getitem__(self, index):
        tile_index, timestamp_index, i, j, chip_size = index

        tile_sample = load_tile(
            self.fs_mapper,
            self.tile_paths[tile_index],
            masks=self.masks,
            revisit=timestamp_index,
        )

        chip_sample = {}
        for key in tile_sample.keys():
            chip_sample[key] = tile_sample[key][
                ..., i : i + chip_size, j : j + chip_size
            ]

        if self.transform:
            chip_sample = self.transform(chip_sample)

        return chip_sample

    def __len__(self):
        return len(self.tile_paths)

    def per_worker_init(self) -> None:
        """
        This is needed to work with gcp and pytorch datamodules.
        Otherwise it won't work when num_workers > 0
        """
        asyn.iothread[0] = None
        asyn.loop[0] = None

        fs = gcsfs.GCSFileSystem(
            access="read_only",
            # skip_instance_cache=True  # Why skip_instance_cache?  See https://github.com/dask/gcsfs/issues/379#issuecomment-839929801
        )
        self.fs_mapper = fs.get_mapper
