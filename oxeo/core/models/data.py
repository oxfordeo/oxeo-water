from datetime import datetime
from functools import partial
from typing import Dict, List, Tuple, Union

import geopandas as gpd
import pyproj
import pystac_client
import sqlalchemy
import stackstac
import xarray as xr
from pyproj import CRS
from shapely import wkb
from sqlalchemy import column, table
from sqlalchemy.sql import select

SearchParamValue = Union[str, list, int, float]
SearchParams = Dict[str, SearchParamValue]

DATE_EARLIEST = datetime(1900, 1, 1)
DATE_LATEST = datetime(2200, 1, 1)


def fetch_water_list(
    water_list: List[int], engine: sqlalchemy.engine.Engine
) -> List[Tuple[int, str, str]]:
    water = table("water", column("area_id"), column("name"), column("geom"))
    with engine.connect() as conn:
        s = select(
            [water.c.area_id, water.c.name, water.c.geom],
            water.c.area_id.in_(water_list),
        )
        data = conn.execute(s).fetchall()

    return data


def data2gdf(
    data: List[Tuple[int, str, str]],
) -> gpd.GeoDataFrame:
    wkb_hex = partial(wkb.loads, hex=True)
    gdf = gpd.GeoDataFrame(data, columns=["area_id", "name", "geometry"])
    gdf.geometry = gdf.geometry.apply(wkb_hex)
    gdf.crs = CRS.from_epsg(4326)
    return gdf


def get_water_geoms(
    water_list: List[int],
    db_user: str,
    db_password: str,
    db_host: str,
) -> gpd.GeoDataFrame:
    """Calls the fetch_water_list and data2gdf in one go."""
    db_url = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:5432/geom"
    engine = sqlalchemy.create_engine(db_url)
    return data2gdf(fetch_water_list(water_list, engine))


def get_aoi_from_stac_catalog(
    catalog_url: str, search_params: SearchParams
) -> xr.DataArray:
    """Get an aoi from a stac catalog using the search params.
    If the aoi is not chunk aligned an offset will be added.

    Args:
        catalog_url (str): the catalog url
        search_params (SearchParams): the search params to be used by pystac_client search.
                    It is mandatory that the search_params contain the 'bbox' key (min_x, min_y, max_x, max_y)

    Returns:
        xr.DataArray: the aoi as an xarray dataarray
    """
    catalog = pystac_client.Client.open(catalog_url)
    items = catalog.search(**search_params).get_all_items()
    stack = stackstac.stack(items, resolution=10)

    min_x_utm, min_y_utm = pyproj.Proj(stack.crs)(
        search_params["bbox"][0], search_params["bbox"][1]
    )
    max_x_utm, max_y_utm = pyproj.Proj(stack.crs)(
        search_params["bbox"][2], search_params["bbox"][3]
    )

    aoi = stack.loc[..., max_y_utm:min_y_utm, min_x_utm:max_x_utm]

    # Sometimes the aoi chunksize is not correctly aligned with the original chunksize.
    # So we add an offset if needed comparing the original
    # chunk size to the new aoi chunk size.
    # We add a buffer in meters to y_top, y_bottom, x_left and x_right.

    m_buffer_y_top = (
        stack.chunksizes["y"][0] - aoi.chunksizes["y"][0]
    ) * stack.resolution
    m_buffer_y_bottom = (
        stack.chunksizes["y"][0] - aoi.chunksizes["y"][-1]
    ) * stack.resolution
    m_buffer_x_left = (
        stack.chunksizes["x"][0] - aoi.chunksizes["x"][0]
    ) * stack.resolution
    m_buffer_x_right = (
        stack.chunksizes["x"][0] - aoi.chunksizes["x"][-1]
    ) * stack.resolution
    aoi = stack.loc[
        ...,
        max_y_utm + m_buffer_y_top : min_y_utm - m_buffer_y_bottom,
        min_x_utm - m_buffer_x_left : max_x_utm + m_buffer_x_right,
    ]
    return aoi
