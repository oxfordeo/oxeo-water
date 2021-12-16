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
        dataset_len = len(self.dataset)
        tile_dates = self.dataset.dates
        for _ in range(self.length):
            # Choose a random tile_index
            tile_index = random.randint(0, dataset_len - 1)

            timestamp_index = random.randint(0, tile_dates[tile_index] - 1)

            # Choose random i and j
            i, j = random.sample(range(self.tile_size - self.chip_size), 2)

            yield tile_index, timestamp_index, i, j, self.chip_size

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
        dataset_len = len(self.dataset)
        tile_dates = self.dataset.dates
        for tile_index in range(dataset_len):
            for timestamp_index in range(tile_dates[tile_index]):
                for i in range(0, self.tile_size, self.chip_size):
                    for j in range(0, self.tile_size, self.chip_size):
                        yield tile_index, timestamp_index, i, j, self.chip_size

    def __len__(self) -> int:
        """Return the number of samples in a single epoch.
        Returns:
            length of the epoch
        """
        chips_per_tile = (self.tile_size // self.chip_size) ** 2
        return len(self.dataset) * chips_per_tile * sum(self.dataset.dates)
