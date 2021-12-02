import itertools
from datetime import datetime
from typing import List, Set, Tuple, Union

import numpy as np
import zarr
from attr import frozen
from satextractor.models import Tile
from shapely.geometry import MultiPolygon, Polygon


@frozen
class TilePath:
    tile: Tile
    constellation: str
    bucket: str = "oxeo-water"
    root: str = "prod"

    @property
    def path(self):
        return f"{self.bucket}/{self.root}/{self.tile.id}/{self.constellation}"


@frozen
class WaterBody:
    area_id: int
    name: str
    geometry: Union[Polygon, MultiPolygon]
    paths: List[TilePath]


@frozen
class TimeseriesMask:
    mask: np.ndarray  # TxHxW
    dates: List[datetime]
    constellation: str
    resolution: int


def parse_xy(tile: Tile) -> Tuple[int, int]:
    return (tile.xloc, tile.yloc)


def get_dates_in_common(
    patch_paths: List[TilePath], constellation: str
) -> List[datetime]:
    dates: Set[str] = set()
    for pp in patch_paths:
        date = zarr.open(f"gs://{pp.path}/timestamps", "r")[:]
        if len(dates) == 0:
            dates = set(date)
        else:
            dates = dates.intersection(date)
    return zarr_dates_to_datetime(dates)


def nearest(items: List[datetime], pivot: datetime):
    return min(items, key=lambda x: abs(x - pivot))


def zarr_dates_to_datetime(dates: Set[str]) -> List[datetime]:
    return [datetime.strptime(x[:10], "%Y-%m-%d") for x in sorted(list(dates))]


def get_patch_size(patch_paths: List[TilePath]) -> int:  # in pixels
    sizes = []
    for patch in patch_paths:
        z = zarr.open(f"gs://{patch.path}/data", "r")
        x, y = z.shape[2:]
        assert x == y, "Must use square patches"
        sizes.append(x)
    assert len(set(sizes)) == 1, "All sizes must be the same"
    return sizes[0]


def get_tile_size(tiles: List[Tile]) -> int:  # in metres
    sizes = []
    for tile in tiles:
        x, y = tile.bbox_size_x, tile.bbox_size_y
        assert x == y, "Must use square tiles"
        sizes.append(x)
    assert len(set(sizes)) == 1, "All sizes must be the same"
    return sizes[0]


date_earliest = datetime(1900, 1, 1)
date_latest = datetime(2200, 1, 1)


def merge_masks_all_constellations(
    waterbody: WaterBody,
    model_name: str,
) -> List[TimeseriesMask]:
    constellations = list({t.constellation for t in waterbody.paths})
    mask_list = [
        merge_masks_one_constellation(waterbody, model_name, constellation)
        for constellation in constellations
    ]
    return mask_list


def merge_masks_one_constellation(
    waterbody: WaterBody,
    model_name: str,
    constellation: str,
    start_date: datetime = date_earliest,
    end_date: datetime = date_latest,
):
    patch_paths: List[TilePath] = [
        pp for pp in waterbody.paths if pp.constellation == constellation
    ]
    xy = [parse_xy(pp.tile) for pp in patch_paths]
    x, y = list(zip(*xy))
    # TODO improve this
    # Hack to deal with changed tile naming system
    xy = [(p[0] - min(x), p[1] - min(y)) for p in xy]
    x, y = list(zip(*xy))
    max_x = max(x)
    max_y = max(y)

    patch_size = get_patch_size(patch_paths)
    tile_size = get_tile_size([pp.tile for pp in patch_paths])
    resolution = int(patch_paths / tile_size)

    common_dates = get_dates_in_common(patch_paths, constellation)
    nearest_start_date = nearest(common_dates, start_date)
    nearest_end_date = nearest(common_dates, end_date)

    start_date_index = common_dates.index(nearest_start_date)
    end_date_index = common_dates.index(nearest_end_date)

    # Create the fullmask array to contain the patches
    full_mask = np.zeros(
        (
            end_date_index - start_date_index,
            (max_y + 1) * patch_size,
            (max_x + 1) * patch_size,
        )
    )

    for i, pp in enumerate(patch_paths):
        # Get the dates for that patch
        dates = zarr.open(f"gs://{pp.path}/timestamps", "r")[:]
        dates = zarr_dates_to_datetime(dates)

        # This is tricky. Here I check for the dates that all patches share
        # I get the indices of those dates for each of the patches. So I can
        # Extract the correct patches at the end.
        date_indices = {d: index for index, d in enumerate(dates) if d in common_dates}

        keys = list(date_indices.keys())
        start_date_index = keys.index(nearest_start_date)
        end_date_index = keys.index(nearest_end_date)

        date_indices = dict(
            itertools.islice(date_indices.items(), start_date_index, end_date_index)
        )
        date_indices_vals = list(date_indices.values())

        # Once I have the indices I can get the patch and append it to the fullmask
        arr = zarr.open(f"gs://{pp.path}/mask/{model_name}", "r")
        start_y = (max_y - xy[i][1]) * patch_size
        end_y = start_y + patch_size
        start_x = xy[i][0] * patch_size
        end_x = start_x + patch_size

        # I need to do this because zarr doesn't support indexing like numpy
        # So I bring all the dates until the last one and then a filter them.
        arr = arr[: date_indices_vals[-1] + 1].astype(np.uint8)
        full_mask[:, start_y:end_y, start_x:end_x] = arr[date_indices_vals, :]
        full_mask = full_mask.astype(np.uint8)

    # Return the mask and tha common dates in the given range.
    return TimeseriesMask(
        mask=full_mask,
        dates=list(date_indices.keys()),
        constellation=constellation,
        resolution=resolution,
    )
