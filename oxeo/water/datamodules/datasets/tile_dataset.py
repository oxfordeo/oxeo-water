from typing import Callable, Iterable, List, Optional

import gcsfs
import zarr
from fsspec import asyn
from joblib import Memory
from loguru import logger
from torch.utils.data import Dataset

from oxeo.satools.io import strdates_to_datetime
from oxeo.water.models.utils import TilePath, load_tile, resize_sample

from .utils import np_index


class TileDataset(Dataset):
    """A dataset that reads from zarr tiles."""

    def __init__(
        self,
        tile_paths: List[TilePath],
        transform: Optional[Callable] = None,
        masks: Iterable[str] = (),
        target_size: int = None,
        bands: Iterable[str] = None,
        cache_dir: str = None,
        cache_bytes: int = None,
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
        self.target_size = target_size
        self.bands = tuple(bands)

        if cache_dir is not None:
            logger.info(f"Using cache_dir {cache_dir}")
            mem = Memory(
                cachedir=cache_dir, verbose=0, mmap_mode="c", bytes_limit=cache_bytes
            )
            self.load_tile_and_resize = mem.cache(self.load_tile_and_resizeload_tile)

        self.tile_dates = {
            tile_path: strdates_to_datetime(
                zarr.open_array(tile_path.timestamps_path)[:]
            )
            for tile_path in self.tile_paths
        }

        self.tiles_ids = [tile_path.tile.id for tile_path in self.tile_paths]

    def load_tile_and_resize(
        self,
        tile_path: TilePath,
        revisit: int = None,
    ):
        sample = load_tile(
            self.fs_mapper,
            tile_path,
            masks=self.masks,
            revisit=revisit,
            bands=self.bands,
        )
        sample = resize_sample(sample, self.target_size)

    def __getitem__(self, index):
        tile_path, timestamp, i, j, chip_size = index

        timestamp_index = np_index(self.tile_dates[tile_path], timestamp)

        tile_sample = self.load_tile_and_resize(tile_path, timestamp_index)

        chip_sample = {}
        for key in tile_sample.keys():
            chip_sample[key] = tile_sample[key][
                ..., i : i + chip_size, j : j + chip_size
            ]

        chip_sample["constellation"] = tile_path.constellation

        if self.transform:
            chip_sample = self.transform(chip_sample)

        return chip_sample

    def __len__(self):
        return len(self.tile_dates)

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
