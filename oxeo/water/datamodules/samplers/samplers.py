import random
from typing import Iterator, Tuple

import attr
from torch.utils.data import Dataset, Sampler

from oxeo.core.logging import logger


@attr.s
class RandomSampler(Sampler):
    dataset: Dataset = attr.ib()
    chip_size: int = attr.ib(converter=int)
    revisits_per_epoch: int = attr.ib(converter=int)
    samples_per_revisit: int = attr.ib(converter=int)
    length: int = attr.ib(init=False)

    def __attrs_post_init__(self):
        self.length = self.revisits_per_epoch * self.samples_per_revisit

    def __iter__(self) -> Iterator[Tuple[int, int, int]]:
        """Return the index of a dataset.
        Returns:
            (tile_index, i, j) coordinates to index a dataset
        """
        tile_dates = self.dataset.tile_dates
        target_size = self.dataset.target_size

        unpacked_tile_dates = [
            (
                tile_path,
                v,
            )
            for tile_path in tile_dates.keys()
            for v in tile_dates[tile_path]
        ]

        tile_revisits_to_use = random.choices(
            unpacked_tile_dates, k=self.revisits_per_epoch
        )

        logger.debug(f"Tile revisits to use: {tile_revisits_to_use}")

        for _ in range(self.length):
            # Choose a random tile_index
            tile_path, timestamp = random.choice(tile_revisits_to_use)

            # Choose random i and j
            i, j = random.choices(range(target_size - self.chip_size + 1), k=2)
            logger.debug(
                f"{tile_path.data_path}, {timestamp}, {i}, {j}, {self.chip_size}"
            )
            yield tile_path, timestamp, i, j, self.chip_size

    def __len__(self) -> int:
        """Return the number of samples in a single epoch.
        Returns:
            length of the epoch
        """
        return self.length


@attr.s
class GridSampler(Sampler):
    dataset: Dataset = attr.ib()
    tile_size: int = attr.ib(converter=int)
    chip_size: int = attr.ib(converter=int)

    def __iter__(self) -> Iterator[Tuple[int, int, int]]:
        """Return the index of a dataset.
        Returns:
            (tile_index, i, j) coordinates to index a dataset
        """

        limit = self.tile_size - self.chip_size

        for tile_id in self.dataset.tile_dates.keys():
            for timestamp in self.dataset.tile_dates[tile_id]:
                for i in range(0, self.tile_size, self.chip_size):
                    for j in range(0, self.tile_size, self.chip_size):
                        if i >= limit:
                            i = limit
                        if j >= limit:
                            j = limit

                        yield tile_id, timestamp, i, j, self.chip_size

    def __len__(self) -> int:
        """Return the number of samples in a single epoch.
        Returns:
            length of the epoch
        """
        chips_per_tile = (self.tile_size // self.chip_size) ** 2
        return (
            len(self.dataset)
            * chips_per_tile
            * sum(len(d) for d in self.dataset.tile_dates.values())
        )
