from datetime import datetime
from functools import partial
from typing import Union

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
SearchParams = dict[str, SearchParamValue]

DATE_EARLIEST = datetime(1900, 1, 1)
DATE_LATEST = datetime(2200, 1, 1)


def fetch_water_list(
    water_list: list[int], engine: sqlalchemy.engine.Engine
) -> list[tuple[int, str, str]]:
    water = table("water", column("area_id"), column("name"), column("geom"))
    with engine.connect() as conn:
        s = select(
            [water.c.area_id, water.c.name, water.c.geom],
            water.c.area_id.in_(water_list),
        )
        data = conn.execute(s).fetchall()

    return data


def data2gdf(
    data: list[tuple[int, str, str]],
) -> gpd.GeoDataFrame:
    wkb_hex = partial(wkb.loads, hex=True)
    gdf = gpd.GeoDataFrame(data, columns=["area_id", "name", "geometry"])
    gdf.geometry = gdf.geometry.apply(wkb_hex)
    gdf.crs = CRS.from_epsg(4326)
    return gdf


def get_water_geoms(
    water_list: list[int],
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
    stack = stackstac.stack(items)

    start_y = stack.y[-1].values
    start_x = stack.x[0].values

    m_y_chunk_size = stack.chunksizes["y"][0] * stack.resolution
    m_x_chunk_size = stack.chunksizes["x"][0] * stack.resolution

    min_x_utm, min_y_utm = pyproj.Proj(stack.crs)(
        search_params["bbox"][0], search_params["bbox"][1]
    )
    max_x_utm, max_y_utm = pyproj.Proj(stack.crs)(
        search_params["bbox"][2], search_params["bbox"][3]
    )

    offset_y_top = (max_y_utm - start_y) % m_y_chunk_size
    offset_y_bottom = m_y_chunk_size - offset_y_top

    offset_x_left = (min_x_utm - start_x) % m_x_chunk_size
    offset_x_right = m_x_chunk_size - offset_x_left

    aoi = stack.loc[
        ...,
        max_y_utm + offset_y_top : min_y_utm - offset_y_bottom,
        min_x_utm - offset_x_left : max_x_utm + offset_x_right,
    ]
    # Sometimes the chunksize is not correctly aligned because of the utm projection.
    # So we check add and add an offset if needed.
    if (
        aoi.chunksizes["y"][0] != stack.chunksizes["y"][0]
        or aoi.chunksizes["x"][0] != stack.chunksizes["x"][0]
    ):
        m_buffer_y = (
            stack.chunksizes["y"][0] - aoi.chunksizes["y"][0]
        ) * stack.resolution
        m_buffer_x = (
            stack.chunksizes["x"][0] - aoi.chunksizes["x"][0]
        ) * stack.resolution
        aoi = stack.loc[
            ...,
            max_y_utm + offset_y_top + m_buffer_y : min_y_utm - offset_y_bottom,
            min_x_utm - offset_x_left - m_buffer_x : max_x_utm + offset_x_right,
        ]
    return aoi
