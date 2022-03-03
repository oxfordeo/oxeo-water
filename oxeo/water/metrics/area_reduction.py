from typing import List, Optional, Tuple, Union
from functools import partial

import dask.array as da
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS
from rasterio import features, transform
from satextractor.models import Tile
from shapely.geometry import MultiPolygon, Polygon
from skimage import morphology

from oxeo.core.logging import logger
from oxeo.core.models.timeseries import TimeseriesMask
from oxeo.core.models.waterbody import WaterBody

"""
The computational graph of this module is as follows:

entrypoing: seg_area_all
- maps across constellations and for each calls: seg_area_single
- - first calls rasterize_geom for an OSM raster for that lake and constellation
- - then maps across years and for each calls: mask_and_get_area
- - - first calls mask_cube, which calls clean_frame and mask_osm_frame
- - - then calls reduce_to_area, which returns a scalar that bubbles up
"""


def seg_area_all(
    segs: List[TimeseriesMask],
    waterbody: WaterBody,
    label_to_mask: int = 1,
) -> pd.DataFrame:
    """Create a scalar timeseries for all tiles in a WaterBody and ALL constellations."""
    logger.info(f"seg_area_all; {waterbody.area_id}: calculate metrics")
    geom = waterbody.geometry
    tiles = [tp.tile for tp in waterbody.paths]
    dfs = [seg_area_single(tsm, tiles, geom, label_to_mask) for tsm in segs]
    return pd.concat(dfs, axis=0)


def seg_area_single(
    tsm: TimeseriesMask,
    tiles: list[Tile],
    geom: Union[Polygon, MultiPolygon],
    label_to_mask: int = 1,
) -> pd.DataFrame:
    """Calculate scalar timeseries for all tiles in waterbody for a SINGLE constellation."""
    logger.info(f"Calulating metrics for {tsm.constellation}")
    # osm_raster must be created separately for each constellation
    # as they have different resolutions
    shape = tsm.data.shape[2:]
    epsg = tiles[0].epsg
    osm_raster = rasterize_geom(geom=geom, tiles=tiles, shape=shape, epsg=epsg)
    logger.info(f"Created OSM mask with {osm_raster.shape=}")

    # Prepare a partial func of mask_and_get_area to groupby().map() with xarray
    masker = partial(
        mask_and_get_area,
        osm_raster=osm_raster,
        label_to_mask=label_to_mask,
        unit="meter",
        resolution=tsm.resolution,
    )
    area = tsm.data.groupby("revisits.year").map(masker)

    df = pd.DataFrame(
        data={
            "date": tsm.data.revisits.compute().data,
            "area": area.compute().data,
            "constellation": tsm.constellation,
        }
    )
    return df


def mask_and_get_area(
    data: xr.DataArray,
    osm_raster: np.ndarray,
    label_to_mask: int,
    unit: str,
    resolution: int,
) -> xr.DataArray:
    """Simple wrapper for mask_cube and reduce_to_area."""
    logger.info(f"Mask and get area for group with {len(data.revisits)=}")
    data = mask_cube(data, osm_raster, label_to_mask)
    area = reduce_to_area(data, unit, resolution=resolution)
    return area


def mask_cube(
    data: xr.DataArray,
    osm_raster: np.ndarray,
    label_to_mask: int = 1,
) -> xr.DataArray:
    """Clean and OSM mask each frame in the cube."""

    # TODO
    # This loops manually because skimage isn't set up for more than 2 dims
    # Should probably use numba here
    arr = data[:, 0, ...].compute().data
    all_masks = [
        mask_osm_frame(clean_frame(arr[i, ...], label_to_mask), osm_raster)
        for i in range(len(data))
    ]
    block = da.stack(all_masks, axis=0).compute()
    # Dropped the 'band' dimension as not needed
    # Might be better to keep it for consistency with other stuff?
    osm_masked = xr.DataArray(block, dims=["revisits", "height", "width"])
    return osm_masked


def clean_frame(
    lab: np.ndarray,
    label_to_mask: int = 1,
) -> np.ndarray:
    """Clean a single timeseries frame.

    Args:
        lab: HxW array
        label_to_mask: the value that indicates True

    Returns
        array with same shape
    """
    lab[lab != label_to_mask] = 0
    lab = lab.astype(bool)
    lab = morphology.closing(lab, morphology.square(3))
    lab = morphology.remove_small_holes(lab, area_threshold=50, connectivity=2)
    lab = morphology.remove_small_objects(lab, min_size=50, connectivity=2)
    lab = morphology.label(lab, background=0, connectivity=2)
    return lab


def mask_osm_frame(
    lab: np.ndarray,
    geom_raster: np.ndarray,
) -> np.ndarray:
    """Mask a single frame to only include pixels that are contiguous with OSM geom.
    See: https://github.com/oxfordeo/oxeo-water/issues/14

    Args:
        lab: HxW array
        geom_raster: array with same dimensions

    Returns:
        array with same shape
    """
    # Overlay the OSM mask and find the label IDs of all
    # labelled water that it covers
    masked = geom_raster * lab
    keepers = [val for val in np.unique(masked) if val != 0]

    # Go back to the labelled OSM mask and keep all pixels with those labels
    fin = np.isin(lab, keepers)
    da_arr = da.from_array(fin)
    return da_arr


def reduce_to_area(
    seg: Union[np.ndarray, xr.DataArray],
    unit: str = "pixel",
    resolution: Optional[int] = 1,
) -> Union[np.ndarray, xr.DataArray]:
    """Calculate the area of each frame in a cube
    Will reduce ndim by 2 (the HxW dimensions) so Nx...xHxW becomes Nx...

    Args:
        seg (np.ndarray): N dimensional binary segmentation.
        unit (str): the unit of the area. Can be in pixels or meters.
        resolution (Optional[int]): if unit is meter the seg resolution must be present

    Returns:
        float: total area (Nx...)
    """

    UNITS = ["pixel", "meter"]
    assert unit in UNITS, f"unit must be one of {UNITS}"

    total_area = (seg > 0).sum(axis=(-2, -1))
    if unit == "meter":
        assert resolution is not None, "resolution is mandatory when unit is 'meter'"
        total_area *= resolution**2

    return total_area


def rasterize_geom(
    geom: Union[Polygon, MultiPolygon],
    tiles: List[Tile],
    shape: Tuple[int, int],
    epsg: int,
) -> np.ndarray:
    """Convert WGS84 Geometry to an array in the specified target CRS (EPSG code).

    Args:
        geom: the geometry
        tiles: the Tiles that the geometry covers
        shape: the target array shape HxW
        epsg: the target CRS EPSG code

    Returns:
        HxW array
    """
    geom = gpd.GeoSeries(geom, crs=CRS.from_epsg(4326)).to_crs(epsg=epsg).geometry
    min_x = min(t.min_x for t in tiles)
    min_y = min(t.min_y for t in tiles)
    max_x = max(t.max_x for t in tiles)
    max_y = max(t.max_y for t in tiles)
    height, width = shape

    affine = transform.from_bounds(
        west=min_x,
        south=min_y,
        east=max_x,
        north=max_y,
        width=width,
        height=height,
    )

    geom_raster = features.rasterize(
        geom,
        out_shape=shape,
        fill=0,
        default_value=1,
        all_touched=True,
        transform=affine,
    )

    return geom_raster
