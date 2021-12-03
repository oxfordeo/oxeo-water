from typing import List, Optional, Tuple, Union

import dask.array as da
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import xarray as xr
from pyproj import CRS
from satextractor.models import Tile
from scipy import stats
from shapely.geometry import MultiPolygon, Polygon
from skimage.morphology import (
    closing,
    label,
    remove_small_holes,
    remove_small_objects,
    square,
)

from oxeo.water.models.utils import TimeseriesMask, WaterBody

UNITS = ["pixel", "meter"]


def segmentation_area_multiple(
    segs: List[TimeseriesMask],
    waterbody: WaterBody,
) -> pd.DataFrame:

    geom = waterbody.geometry
    tiles = [tp.tile for tp in waterbody.paths]

    dfs = []
    for tsm in segs:
        print(f"Calulating metrics for {tsm.constellation}")
        # osm_raster must be created separately for each constellation
        # as they have different resolutions
        shape = tsm.data.shape[2:]
        epsg = tiles[0].epsg
        osm_raster = rasterize_geom(geom=geom, tiles=tiles, shape=shape, epsg=epsg)
        print(f"Created OSM mask with {osm_raster.shape=}")

        data = mask_cube(tsm.data, osm_raster)
        print(f"Masked data cube with {data.shape=}")

        area = segmentation_area(data, unit="meter", resolution=tsm.resolution)
        print(f"Calculated 1D area array with {area.shape=}")

        df = pd.DataFrame(
            data={
                "date": tsm.data.revisits.compute().data,
                "area": area.compute().data,
                "constellation": tsm.constellation,
            }
        )
        dfs.append(df)
        print("Inserted into DF")
    return pd.concat(dfs, axis=0)


def mask_single(arr: da.Array, i: int, geom_raster: np.ndarray):
    # TODO will need to modify this once the incoming masks
    # have multiple classes (land/water/cloud etc)
    # Basically must just set all non-water to 0 and all water to 1
    lab = arr[i, 0, ...].compute().astype(bool)
    lab = closing(lab, square(3))
    lab = remove_small_holes(lab, area_threshold=50, connectivity=2)
    lab = remove_small_objects(lab, min_size=50, connectivity=2)
    lab = label(lab, background=0, connectivity=2)

    # Overlay the OSM mask and find the label IDs of all
    # labelled water that it covers
    masked = geom_raster * lab
    keepers = [val for val in np.unique(masked) if val != 0]

    # Go back to the labelled OSM mask and keep all pixels with those labels
    fin = np.isin(lab, keepers)
    da_arr = da.from_array(fin)
    return da_arr


def mask_cube(data: xr.DataArray, osm_raster: np.ndarray) -> xr.DataArray:
    # TODO Probably tere's some clever Dasky stuff to do here
    # Right now it's just a sequential loop
    all_masks = [mask_single(data, i, osm_raster) for i in range(len(data))]
    block = da.stack(all_masks, axis=0)
    # Dropped the 'band' dimension as not needed
    # Might be better to keep it for consistency with other stuff?
    osm_masked = xr.DataArray(block, dims=["revisits", "height", "width"])
    return osm_masked


def segmentation_area(
    seg: Union[np.ndarray, xr.DataArray],
    unit: str = "pixel",
    resolution: Optional[int] = 1,
) -> Union[np.ndarray, xr.DataArray]:
    """Get the total area of a segmentation (Nx..xHxW)

    Args:
        seg (np.ndarray): N dimensional binary segmentation.
        unit (str): the unit of the area. Can be in pixels or meters.
        resolution (Optional[int]): if unit is meters the seg resolution must be present

    Returns:
        float: total area (Nx...)
    """
    assert unit in UNITS, f"unit must be one of {UNITS}"
    total_area = (seg > 0).sum(axis=(-2, -1))

    if unit == "meter":
        assert resolution is not None, "resolution is mandatory when unit is 'meter'"
        total_area *= resolution ** 2

    return total_area


def pearson(x, y):
    return stats.pearsonr(x, y)[0]


def rasterize_geom(
    geom: Union[Polygon, MultiPolygon],
    tiles: List[Tile],
    shape: Tuple[int, int],
    epsg: int,
) -> np.ndarray:
    geom = gpd.GeoSeries(geom, crs=CRS.from_epsg(4326)).to_crs(epsg=epsg).geometry
    min_x = min(t.min_x for t in tiles)
    min_y = min(t.min_y for t in tiles)
    max_x = max(t.max_x for t in tiles)
    max_y = max(t.max_y for t in tiles)
    height, width = shape

    affine = rio.transform.from_bounds(
        west=min_x,
        south=min_y,
        east=max_x,
        north=max_y,
        width=width,
        height=height,
    )

    geom_raster = rio.features.rasterize(
        geom,
        out_shape=shape,
        fill=0,
        default_value=1,
        all_touched=True,
        transform=affine,
    )

    return geom_raster
