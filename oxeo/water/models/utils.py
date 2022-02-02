from datetime import datetime
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import xarray as xr
import zarr
from attr import define
from loguru import logger
from pystac.extensions.eo import Band
from satextractor.models import Tile
from satextractor.models.constellation_info import BAND_INFO
from shapely.geometry import MultiPolygon, Polygon
from torchvision.transforms.functional import InterpolationMode, resize
from zarr.core import ArrayNotFoundError

from oxeo.satools.io import ConstellationData, constellation_dataarray


def get_band_list(constellation: str) -> List[str]:
    BAND_INFO["sentinel-1"] = {
        "B1": {"band": Band.create(name="B1", common_name="VV")},
        "B2": {"band": Band.create(name="B2", common_name="VH")},
    }
    return [b["band"].common_name for b in BAND_INFO[constellation].values()]


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

    @property
    def metadata_path(self):
        return f"{self.path}/metadata"


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
        logger.info(f"Loading to check size {arr_path=}")
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
    waterbody: WaterBody, mask: str
) -> List[TimeseriesMask]:
    constellations = list({t.constellation for t in waterbody.paths})
    mask_list = []
    for constellation in constellations:
        try:
            logger.info(
                f"merge_masks_all_constellations; {constellation=}; {waterbody.area_id=}: merging"
            )
            tsm = merge_masks_one_constellation(waterbody, constellation, mask)
            mask_list.append(tsm)
        except (ValueError, FileNotFoundError, KeyError, ArrayNotFoundError) as e:
            logger.info(f"Failed to load {constellation=} on {waterbody.area_id=}, error {e}")
            logger.info("Continuing with other constellations")
    return mask_list


def merge_masks_one_constellation(
    waterbody: WaterBody,
    constellation: str,
    mask: str,
):
    patch_paths: List[TilePath] = [
        pp for pp in waterbody.paths if pp.constellation == constellation
    ]
    if len(patch_paths) == 0:
        raise ValueError(f"Constellation '{constellation}' not found in waterbody")

    logger.info(
        f"merge_masks_one_constellation; {constellation=}; {waterbody.area_id}: get details"
    )
    tile_size = get_tile_size([pp.tile for pp in patch_paths])  # in metres
    patch_size = get_patch_size(patch_paths)  # in pixels
    resolution = int(tile_size / patch_size)
    root_dir = patch_paths[0].root

    tile_ids = [
        tp.tile.id for tp in waterbody.paths if tp.constellation == constellation
    ]
    paths = [f"gs://oxeo-water/{root_dir}/{t}" for t in tile_ids]
    c_data = ConstellationData(constellation, bands=["mask"], paths=paths)

    logger.info(
        f"merge_masks_one_constellation; {constellation=}; {waterbody.area_id}: create dataarray"
    )
    data_arr = constellation_dataarray(c_data, data_path=f"mask/{mask}")

    return TimeseriesMask(
        data=data_arr,
        constellation=constellation,
        resolution=resolution,
    )


def load_tile(
    fs_mapper,
    tile_path: TilePath,
    masks: Tuple[str, ...] = (),
    revisit: slice = None,
    bands: Tuple[str, ...] = None,
) -> torch.Tensor:
    if bands is not None:
        band_common_names = get_band_list(tile_path.constellation)
        band_indices = [band_common_names.index(b) for b in bands]
    else:
        band_common_names = get_band_list(tile_path.constellation)
        band_indices = list(range(0, len(band_common_names)))

    sample = {}
    arr = zarr.open_array(fs_mapper(tile_path.data_path), mode="r")
    logger.info(f"{arr.shape=}; {revisit=}, {band_indices=}")
    arr = arr.oindex[revisit, band_indices].astype(np.int16)
    logger.info(f"{arr.shape=}")
    for mask in masks:
        mask_arr = zarr.open_array(
            fs_mapper(f"{tile_path.mask_path}/{mask}"), mode="r"
        )[revisit].astype(np.int8)
        mask_arr = mask_arr[np.newaxis, ...]
        sample[mask] = torch.as_tensor(mask_arr)

    sample["image"] = torch.as_tensor(arr)

    return sample


def resize_sample(
    sample: Union[torch.Tensor, Dict[str, torch.Tensor]],
    target_size: int = None,
    interpolation=InterpolationMode.NEAREST,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """Resize sample to target

    Args:
        sample (Union[torch.Tensor, Dict[str, torch.Tensor]]): Can be a tensor or dict of tensors
        target_size (int, optional): Target size. Defaults to None.
        interpolation ([type], optional): Only used when sample is torch.Tensor.
                                          Defaults to InterpolationMode.NEAREST.

    Returns:
        torch.Tensor: the resampled tensor or dict of tensors
    """
    logger.debug(f"Resizing sample to {target_size}")
    if target_size is not None:
        if isinstance(sample, dict):
            resized_sample = {}
            for key in sample.keys():
                if key == "image":
                    resized_sample[key] = resize(
                        sample[key], target_size, InterpolationMode.BILINEAR
                    )
                else:
                    resized_sample[key] = resize(
                        sample[key], target_size, InterpolationMode.NEAREST
                    )
        elif isinstance(sample, torch.Tensor):
            resized_sample = resize(sample, target_size, interpolation)
    return resized_sample


def load_tile_and_resize(
    fs_mapper,
    tile_path: TilePath,
    masks,
    revisit: int = None,
    bands=None,
    target_size=None,
):
    sample = load_tile(
        fs_mapper,
        tile_path,
        masks=masks,
        revisit=revisit,
        bands=bands,
    )
    sample = resize_sample(sample, target_size)
    return sample
