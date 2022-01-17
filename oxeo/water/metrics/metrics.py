from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple, Union

import dask.array as da
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from joblib import Parallel, delayed
from loguru import logger
from pyproj import CRS
from rasterio import features, transform
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
from tqdm import tqdm
from zarr.errors import PathNotFoundError

from oxeo.water.models.utils import (
    TilePath,
    TimeseriesMask,
    WaterBody,
    load_tile,
    tile_from_id,
)
from oxeo.water.utils import tqdm_joblib

UNITS = ["pixel", "meter"]


def get_segmentation_area(tsm, tiles, geom, label_to_mask):
    logger.info(f"Calulating metrics for {tsm.constellation}")
    # osm_raster must be created separately for each constellation
    # as they have different resolutions
    shape = tsm.data.shape[2:]
    epsg = tiles[0].epsg
    osm_raster = rasterize_geom(geom=geom, tiles=tiles, shape=shape, epsg=epsg)
    logger.info(f"Created OSM mask with {osm_raster.shape=}")
    data = mask_cube(tsm.data, osm_raster, label_to_mask)
    logger.info(f"Masked data cube with {data.shape=}")

    area = segmentation_area(data, unit="meter", resolution=tsm.resolution)
    logger.info(f"Calculated 1D area array with {area.shape=}")

    df = pd.DataFrame(
        data={
            "date": tsm.data.revisits.compute().data,
            "area": area.compute().data,
            "constellation": tsm.constellation,
        }
    )
    return df


def segmentation_area_multiple(
    segs: List[TimeseriesMask],
    waterbody: WaterBody,
    label_to_mask: int = 1,
    n_jobs=-1,
    verbose=0,
) -> pd.DataFrame:

    geom = waterbody.geometry
    tiles = [tp.tile for tp in waterbody.paths]

    with tqdm_joblib(
        tqdm(
            desc="parallel calculating metrics for constellations.",
            total=len(segs),
        ),
    ):
        dfs = Parallel(n_jobs=n_jobs, verbose=verbose)(
            [
                delayed(get_segmentation_area)(tsm, tiles, geom, label_to_mask)
                for tsm in segs
            ],
        )

    return pd.concat(dfs, axis=0)


def mask_single(arr: da.Array, i: int, label_to_mask: int = 1):
    lab = arr[i, 0, ...].compute()
    lab = lab.where(lab != label_to_mask, 0)
    lab = lab.astype(bool)
    lab = closing(lab, square(3))
    lab = remove_small_holes(lab, area_threshold=50, connectivity=2)
    lab = remove_small_objects(lab, min_size=50, connectivity=2)
    lab = label(lab, background=0, connectivity=2)
    return lab


def overlay_osm(lab: np.ndarray, geom_raster: np.ndarray):
    # Overlay the OSM mask andn   find the label IDs of all
    # labelled water that it covers
    masked = geom_raster * lab
    keepers = [val for val in np.unique(masked) if val != 0]

    # Go back to the labelled OSM mask and keep all pixels with those labels
    fin = np.isin(lab, keepers)
    da_arr = da.from_array(fin)
    return da_arr


def mask_cube(
    data: xr.DataArray, osm_raster: np.ndarray, label_to_mask: int = 1
) -> xr.DataArray:
    # TODO Probably tere's some clever Dasky stuff to do here
    # Right now it's just a sequential loop
    all_masks = [
        overlay_osm(mask_single(data, i, label_to_mask), osm_raster)
        for i in range(len(data))
    ]
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


def tile_stat_per_band(
    fs_mapper,
    tile_ids: List[str],
    bands: List[str],
    constellations: List[str],
    revisits_batch: int = 32,
    stat: Callable = None,
) -> Dict[str, Dict[str, float]]:
    """Calculate the tiles mean per band for every listed constellation

    Args:
        tile_ids (List[str]): The tile ids to include in the mean
        bands (List[str]): The bands to calculate the mean
        constellations (List[str]): The constellations.
        revisits_batch (int): how many revisits to load per step when loading tile.
                                reduce to avoid memory consumption.
        stat (Callable): function to run on tile. Should be a np callable (ie np.mean)

    Returns:
        Dict[str, Dict[str, float]]: A dictionary with each constellation as key, and
                    a dictionary with band_name:value as value
    """
    res = defaultdict(list)
    for tile_id in tqdm(tile_ids):
        for constellation in constellations:
            tile = tile_from_id(tile_id)
            tile_path = TilePath(tile, constellation)
            try:
                zarr_arr = zarr.open(tile_path.data_path, "r")
                revisits = zarr_arr.shape[0]
                for i in range(0, revisits, revisits_batch):
                    tile_tensor = load_tile(
                        fs_mapper=fs_mapper,
                        tile_path=tile_path,
                        revisit=slice(i, i + revisits_batch),
                        bands=bands,
                    )
                    tile_mean = stat(tile_tensor["image"].numpy(), axis=(2, 3))
                    res[constellation].extend(tile_mean.tolist())
            except PathNotFoundError:
                logger.error("Path {tile_path.data_path} not found.")

    for key in res.keys():
        res[key] = np.array(res[key]).mean(0)

    return {
        c: {bands[i]: values[i] for i, _ in enumerate(values)}
        for c, values in res.items()
    }
