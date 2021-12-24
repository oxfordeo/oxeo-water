import random
from itertools import islice
from typing import Callable, Iterable, List, Optional

import gcsfs
import numpy as np
import zarr
from fsspec import asyn
from joblib import Memory
from loguru import logger
from torch.utils.data import IterableDataset, get_worker_info

from oxeo.satools.io import strdates_to_datetime
from oxeo.water.models.utils import TilePath, load_tile

from .utils import np_index


class IterableTileDataset(IterableDataset):
    """A dataset that reads from zarr tiles."""

    def __init__(
        self,
        tile_paths: List[TilePath],
        transform: Optional[Callable] = None,
        masks: Iterable[str] = (),
        target_size: int = None,
        chip_size: int = None,
        bands: Iterable[str] = None,
        revisits_per_epoch: int = None,
        samples_per_revisit: 100 = None,
        sampler: str = "random",
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
        self.chip_size = chip_size
        self.bands = tuple(bands)
        self.revisits_per_epoch = revisits_per_epoch
        self.samples_per_revisit = samples_per_revisit
        self.sampler = sampler

        if cache_dir is not None:
            logger.info(f"Using cache_dir {cache_dir}")
            mem = Memory(
                cachedir=cache_dir, verbose=0, mmap_mode="c", bytes_limit=cache_bytes
            )
            self.load_tile = mem.cache(load_tile)
        else:
            logger.info(f"Not using cache_dir {cache_dir}. Training may be slow.")
            self.load_tile = load_tile

        self.tile_dates = {
            tile_path: strdates_to_datetime(
                zarr.open_array(tile_path.timestamps_path)[:]
            )
            for tile_path in self.tile_paths
        }

    def __iter__(self):
        """Each worker will use self.revisits_per_epoch // worker_info.num_workers
        tiles and extract from them, *samples_per_revisits* chips.
        Each loaded tile will be cached using load_tile function, so subsequent calls will
        load data from memory.
        Yield return a random chip.

        Yields:
            [type]: [description]
        """
        worker_info = get_worker_info()
        tile_dates = self.tile_dates
        unpacked_tile_dates = [
            (
                tile_path,
                v,
            )
            for tile_path in tile_dates.keys()
            for v in tile_dates[tile_path]
        ]

        if self.sampler == "random":

            revisits_per_worker = self.revisits_per_epoch // worker_info.num_workers

            tile_revisits_to_use = random.choices(
                unpacked_tile_dates, k=revisits_per_worker
            )

            indices = []
            for tile_path, timestamp in tile_revisits_to_use:
                i_samples = random.choices(
                    range(self.target_size - self.chip_size + 1),
                    k=self.samples_per_revisit,
                )
                j_samples = random.choices(
                    range(self.target_size - self.chip_size + 1),
                    k=self.samples_per_revisit,
                )
                for n in range(len(i_samples)):
                    indices.append((tile_path, timestamp, i_samples[n], j_samples[n]))
            random.shuffle(indices)
        elif self.sampler == "grid":
            tile_revisits_to_use = unpacked_tile_dates
            limit = self.target_size - self.chip_size
            indices = []
            for tile_path, timestamp in tile_revisits_to_use:
                for i in range(0, self.target_size, self.chip_size):
                    for j in range(0, self.target_size, self.chip_size):
                        if i >= limit:
                            i = limit
                        if j >= limit:
                            j = limit
                        indices.append((tile_path, timestamp, i, j))
            indices = islice(indices, worker_info.id, None, worker_info.num_workers)
        for tile_path, timestamp, i, j in np.array(indices):
            timestamp_index = np_index(self.tile_dates[tile_path], timestamp)

            tile_sample = self.load_tile(
                self.fs_mapper,
                tile_path,
                masks=self.masks,
                revisit=timestamp_index,
                target_size=self.target_size,
                bands=self.bands,
            )

            chip_sample = {}
            for key in tile_sample.keys():
                chip_sample[key] = tile_sample[key][
                    ..., i : i + self.chip_size, j : j + self.chip_size
                ]

            chip_sample["constellation"] = tile_path.constellation

            if self.transform:
                chip_sample = self.transform(chip_sample)

            del tile_sample
            yield chip_sample

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
