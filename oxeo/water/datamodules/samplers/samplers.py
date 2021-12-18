import random
from typing import Iterator, Tuple

import attr
from torch.utils.data import Dataset, Sampler


@attr.s
class RandomSampler(Sampler):
    dataset: Dataset = attr.ib()
    tile_size: int = attr.ib(converter=int)
    chip_size: int = attr.ib(converter=int)
    length: int = attr.ib(converter=int)

    def __iter__(self) -> Iterator[Tuple[int, int, int]]:
        """Return the index of a dataset.
        Returns:
            (tile_index, i, j) coordinates to index a dataset
        """
        tile_dates = self.dataset.tile_dates
        for _ in range(self.length):
            # Choose a random tile_index
            tile_id = random.choice(list(tile_dates.keys()))

            timestamp = random.choice(tile_dates[tile_id])

            # Choose random i and j
            i, j = random.choices(range(self.tile_size - self.chip_size + 1), k=2)

            yield tile_id, timestamp, i, j, self.chip_size

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
