from datetime import datetime
from typing import Callable, Dict, Iterable, List, Optional

import gcsfs
import zarr
from fsspec import asyn
from joblib import Memory
from torch.utils.data import Dataset
from tqdm import tqdm

from oxeo.core.logging import logger
from oxeo.core.models.tile import TilePath
from oxeo.satools.io import strdates_to_datetime
from oxeo.water.models.tile_utils import load_tile_as_dict_and_resize

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
        start_date: str = "0001-01-01",
        end_date: str = "9999-01-01",
        valid_dates: Optional[Dict[str, Dict[str, List[str]]]] = None,
    ):
        """Tile dataset

        Args:
            tile_paths (List[TilePath]): The list of TilePaths to be included in the dataset
            transform (Optional[Callable], optional): Transforms to apply to samples. Defaults to None.
            masks (Iterable[str], optional): Masks to load in the sample (ex: (pekel,cloud_mask)). Defaults to ().
            target_size (int, optional): The target size of the sample. Samples will be rescaled to target. Defaults to None.
            bands (Iterable[str], optional): Bands to load. Defaults to None.
            cache_dir (str, optional): A cache dir in local disk to store load_tile_as_dict function. Defaults to None.
            cache_bytes (int, optional): How many bytes to use as cache in local disk. Defaults to None.
            start_date (str, optional): Dataset will use only dates after (and included) start_date (%Y-%m-%d). Defaults to 0001-01-01.
            end_date (str, optional): Dataset will use only dates before (and included) end_date (%Y-%m-%d). Defaults to 9999-01-01.
            valid_dates (Optional[Dict[str, Dict[str,List[str]]]], optional): A list of valid dates. Only uses the dates in the list. Defaults to None.
        """
        super().__init__()

        self.tile_paths = tile_paths
        self.transform = transform
        self.masks = masks
        self.fs_mapper = None
        self.target_size = target_size
        self.bands = tuple(bands)
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.valid_dates = valid_dates

        if cache_dir is not None:
            logger.info(f"Using cache_dir {cache_dir}")
            mem = Memory(
                cachedir=cache_dir, verbose=0, mmap_mode="c", bytes_limit=cache_bytes
            )
            self.load_tile_as_dict_and_resize = mem.cache(load_tile_as_dict_and_resize)
        else:
            self.load_tile_as_dict_and_resize = load_tile_as_dict_and_resize

        logger.info("Loading all dates for all tiles.")

        self.all_tile_dates = {
            tile_path: [
                d
                for d in strdates_to_datetime(
                    zarr.open_array(tile_path.timestamps_path)[:]
                )
            ]
            for tile_path in tqdm(self.tile_paths)
        }

        self.valid_tile_dates = {
            tile_path: [
                d
                for d in self.all_tile_dates[tile_path]
                if (d in self.get_tile_valid_dates(tile_path))
            ]
            for tile_path in tqdm(self.tile_paths)
        }

        self.tiles_ids = [tile_path.tile.id for tile_path in self.tile_paths]

    def get_tile_valid_dates(self, tile_path):
        if self.valid_dates is None:
            dates = []
            for n in range(int((self.end_date - self.start_date).days) + 1):
                dates.append(self.start_date + datetime.timedelta(n))
            return dates
        else:
            return [
                datetime.strptime(d, "%Y-%m-%d")
                for d in self.valid_dates[tile_path.constellation][tile_path.tile.id]
            ]

    def __getitem__(self, index):
        tile_path, timestamp, i, j, chip_size = index
        timestamp_index = np_index(self.all_tile_dates[tile_path], timestamp)
        tile_sample = self.load_tile_as_dict_and_resize(
            self.fs_mapper,
            tile_path,
            self.masks,
            timestamp_index,
            self.bands,
            self.target_size,
        )

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
