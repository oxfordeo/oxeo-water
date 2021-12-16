import functools
from datetime import datetime
from typing import List, Tuple, Union

import numpy as np
import xarray as xr
import zarr
from attr import define
from satextractor.models import Tile
from shapely.geometry import MultiPolygon, Polygon
from zarr.core import ArrayNotFoundError

from oxeo.satools.io import ConstellationData, constellation_dataarray


def tile_from_id(id):
    zone, row, bbox_size_x, xloc, yloc = id.split("_")
    bbox_size_x, xloc, yloc = int(bbox_size_x), int(xloc), int(yloc)
    bbox_size_y = bbox_size_x
    min_x = xloc * bbox_size_x
    min_y = yloc * bbox_size_y
    max_x = min_x + bbox_size_x
    max_y = min_y + bbox_size_y
    return Tile(
        zone=zone,
        row=row,
        min_x=min_x,
        min_y=min_y,
        max_x=max_x,
        max_y=max_y,
        epsg="NONE",
    )


@define(frozen=True)
class TilePath:
    tile: Tile
    constellation: str
    bucket: str = "oxeo-water"
    root: str = "prod"

    @property
    def path(self):
        return f"gs://{self.bucket}/{self.root}/{self.tile.id}/{self.constellation}"

    @property
    def timestamps_path(self):
        return f"{self.path}/timestamps"

    @property
    def data_path(self):
        return f"{self.path}/data"

    @property
    def mask_path(self):
        return f"{self.path}/mask"


@define
class WaterBody:
    area_id: int
    name: str
    geometry: Union[Polygon, MultiPolygon]
    paths: List[TilePath]


@define
class TimeseriesMask:
    data: xr.DataArray  # TxBxHxW
    constellation: str
    resolution: int


def get_patch_size(patch_paths: List[TilePath]) -> int:  # in pixels
    # TODO: Probably unnecessary to load all patches for this,
    # could just assume they're the same size
    sizes = []
    for patch in patch_paths:
        arr_path = f"gs://{patch.path}/data"
        print(f"Loading to check size {arr_path=}")
        z = zarr.open(arr_path, "r")
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
    mask_list = []
    for constellation in constellations:
        try:
            tsm = merge_masks_one_constellation(waterbody, model_name, constellation)
            mask_list.append(tsm)
        except (ValueError, FileNotFoundError, KeyError, ArrayNotFoundError) as e:
            print(f"Failed to load {constellation=} on {waterbody.area_id=}, error {e}")
            print("Continuing with other constellations")
    return mask_list


def merge_masks_one_constellation(
    waterbody: WaterBody,
    constellation: str,
):
    patch_paths: List[TilePath] = [
        pp for pp in waterbody.paths if pp.constellation == constellation
    ]
    if len(patch_paths) == 0:
        raise ValueError(f"Constellation '{constellation}' not found in waterbody")

    tile_size = get_tile_size([pp.tile for pp in patch_paths])  # in metres
    patch_size = get_patch_size(patch_paths)  # in pixels
    resolution = int(tile_size / patch_size)

    tile_ids = [
        tp.tile.id for tp in waterbody.paths if tp.constellation == constellation
    ]
    paths = [f"gs://oxeo-water/prod/{t}" for t in tile_ids]
    c_data = ConstellationData(constellation, bands=["mask"], paths=paths)
    data_arr = constellation_dataarray(c_data, data_path="mask/pekel")

    return TimeseriesMask(
        data=data_arr,
        constellation=constellation,
        resolution=resolution,
    )


@functools.lru_cache(maxsize=5)
def load_tile(tile_path: TilePath, masks: Tuple[str, ...] = ()):
    sample = {}
    arr = zarr.open_array(tile_path.data_path, mode="r")[:]

    for mask in masks:
        mask_arr = zarr.open_array(f"{tile_path.mask_path}/{mask}", mode="r")[:]
        assert (
            mask_arr.shape[0] == arr.shape[0]
        ), "Image arr and mask timestamps don't match"
        mask_arr = mask_arr[:, np.newaxis, ...]
        sample[mask] = mask_arr

    sample["image"] = arr
    return sample
