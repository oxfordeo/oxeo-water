import csv
from datetime import datetime
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import geopandas as gpd
import pandas as pd
import pyproj
import pystac_client
import rasterio
import requests
import sqlalchemy
import stackstac
import xarray as xr
from pyproj import CRS
from rasterio.windows import Window
from shapely import wkb
from sqlalchemy import column, table
from sqlalchemy.sql import select

SearchParamValue = Union[str, list, int, float]
SearchParams = Dict[str, SearchParamValue]

DATE_EARLIEST = datetime(1900, 1, 1)
DATE_LATEST = datetime(2200, 1, 1)


class CusotomSentinel1Reader:
    """Custom Sentinel 1 reader to avoid errors with stackstac"""

    def __init__(self, url, **kwargs):
        self.url = url

    def read(self, window: Window, **kwargs):
        with rasterio.open(self.url) as src:
            result = src.read(window=window, masked=True, **kwargs)

        return result


def query_asf(
    platform: str = "Sentinel-1A,Sentinel-1B",
    processing_level: str = "GRD_HD",
    beam_mode: str = "IW",
    intersects_with_wkt: str = None,
    start: str = None,
    end: str = None,
) -> pd.DataFrame:
    """Query asf for S1 data.

    Args:
        platform (str, optional): The platform to query. Defaults to "Sentinel-1A,Sentinel-1B".
        processing_level (str, optional): SLC, RAW or GRD_HD. Defaults to "GRD_HD".
        beam_mode (str, optional): S1 beam mode. Defaults to "IW".
        intersects_with_wkt (str, optional): POLYGON in wkt. Defaults to None.
        start (str, optional): start date. Defaults to None.
        end (str, optional): end date. Defaults to None.

    Returns:
        _type_: DataFrame containing asf query results .
    """
    params = dict(
        output="csv",
        platform=platform,
        processingLevel=processing_level,
        beamMode=beam_mode,
        intersectsWith=intersects_with_wkt,
        start=start,
        end=end,
    )
    asf_baseurl = "https://api.daac.asf.alaska.edu/services/search/param?"
    for k, v in params.items():
        asf_baseurl = asf_baseurl + f"{k}={v}&"
    r = requests.post(asf_baseurl)
    reader = csv.DictReader(r.text.splitlines())
    rows = list(reader)
    df = pd.DataFrame(rows)
    return df


def asf_granule_to_aws(asf_query_row: pd.Series) -> str:
    """Converts asf query result row to aws s3 path url for the same granule.

    Args:
        asf_query_row (pd.Series): resulting row from asf query.

    Returns:
        str: the s3 path url
    """
    date = asf_query_row["Acquisition Date"]
    year = date[:4]
    month = int(date[5:7])
    day = int(date[8:10])
    platform = asf_query_row["Granule Name"][:3]
    beam_mode = asf_query_row["Beam Mode"]
    polarization = asf_query_row["GroupID"][6:8]
    granule_split = asf_query_row["Granule Name"].split("_")[4:]
    start_date = granule_split[0]
    end_date = granule_split[1]

    product_id = "_".join(granule_split[2:])
    return f"s3://sentinel-s1-l1c/GRD/{year}/{month}/{day}/{beam_mode}/{polarization}/{platform}_IW_GRDH_1S{polarization}_{start_date}_{end_date}_{product_id}"


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
    catalog_url: str,
    search_params: SearchParams,
    chunk_aligned: bool = False,
    resolution: Optional[int] = None,
) -> xr.DataArray:
    """Get an aoi from a stac catalog using the search params.
    If the aoi is not chunk aligned an offset will be added.

    Args:
        catalog_url (str): the catalog url
        search_params (SearchParams): the search params to be used by pystac_client search.
                    It is mandatory that the search_params contain the 'bbox' key (min_x, min_y, max_x, max_y)
        chunk_aligned (bool): if True the data is chunk aligned
        resolution (Optional[int]): The resolution of the data. If it cannot be infered by stac

    Returns:
        xr.DataArray: the aoi as an xarray dataarray
    """
    catalog = pystac_client.Client.open(catalog_url)
    items = catalog.search(**search_params).get_all_items()
    stack = stackstac.stack(items, resolution=resolution)

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
    if chunk_aligned:
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
