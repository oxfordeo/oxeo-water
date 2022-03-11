from typing import List

import xarray as xr
from attr import define
from zarr.core import ArrayNotFoundError

from oxeo.core.logging import logger
from oxeo.core.models.tile import TilePath, get_patch_size, get_tile_size
from oxeo.core.models.waterbody import WaterBody
from oxeo.satools.io import ConstellationData, constellation_dataarray


@define
class TimeseriesMask:
    data: xr.DataArray  # TxBxHxW
    constellation: str
    resolution: int


@define
class TimeseriesScalar:
    data: xr.DataArray  # T
    constellation: str
    resolution: int


def merge_masks_all_constellations(
    waterbody: WaterBody,
    mask: str,
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
            logger.info(
                f"Failed to load {constellation=} on {waterbody.area_id=}, error {e}"
            )
            logger.info("Continuing with other constellations")
    assert len(mask_list) > 0, f"No constellations were found with mask {mask}"
    return mask_list


def merge_masks_one_constellation(
    waterbody: WaterBody,
    constellation: str,
    mask: str,
) -> TimeseriesMask:
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
    paths = [f"{root_dir}/{t}" for t in tile_ids]
    logger.info(f"paths {paths}")
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
