import csv
import warnings
from datetime import datetime
from functools import partial
from typing import Dict, List, Tuple, Union

import geopandas as gpd
import pandas as pd
import pystac
import pystac_client
import rasterio
import rasterio as rio
import requests
import sqlalchemy
import stackstac
import xarray as xr
from pyproj import CRS
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from sentinelhub import BBox, DataCollection, SentinelHubCatalog
from shapely import wkb
from sqlalchemy import column, table
from sqlalchemy.sql import select
from stackstac.nodata_reader import NodataReader, exception_matches
from stackstac.rio_reader import (
    MULTITHREADED_DRIVER_ALLOWLIST,
    AutoParallelRioReader,
    SelfCleaningDatasetReader,
    SingleThreadedRioDataset,
    ThreadLocalRioDataset,
    ThreadsafeRioDataset,
    _curthread,
)
from stackstac.timer import time

from oxeo.core.logging import logger
from oxeo.core.stac import landsat, sentinel1

SearchParamValue = Union[str, list, int, float]
SearchParams = Dict[str, SearchParamValue]

DATE_EARLIEST = datetime(1900, 1, 1)
DATE_LATEST = datetime(2200, 1, 1)


class AutoParallelRioReaderWithCrs(AutoParallelRioReader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _open(self) -> ThreadsafeRioDataset:
        with self.gdal_env.open:
            with time(f"Initial read for {self.url!r} on {_curthread()}: {{t}}"):
                try:
                    ds = SelfCleaningDatasetReader(
                        rio.parse_path(self.url), sharing=False
                    )
                except Exception as e:
                    msg = f"Error opening {self.url!r}: {e!r}"
                    if exception_matches(e, self.errors_as_nodata):
                        warnings.warn(msg)
                        return NodataReader(
                            dtype=self.dtype, fill_value=self.fill_value
                        )

                    raise RuntimeError(msg) from e
            if ds.count != 1:
                ds.close()
                raise RuntimeError(
                    f"Assets must have exactly 1 band, but file {self.url!r} has {ds.count}. "
                    "We can't currently handle multi-band rasters (each band has to be "
                    "a separate STAC asset), so you'll need to exclude this asset from your analysis."
                )

            # Only make a VRT if the dataset doesn't match the spatial spec we want
            gcps, gcp_crs = ds.gcps

            ds_dict = {
                "crs": gcp_crs.to_epsg(),
                "transform": rasterio.transform.from_gcps(gcps),
                "height": ds.height,
                "width": ds.width,
            }
            if self.spec.vrt_params != ds_dict:
                with self.gdal_env.open_vrt:
                    vrt = WarpedVRT(
                        src_dataset=ds,
                        src_crs=ds_dict["crs"],
                        src_transform=ds_dict["transform"],
                        sharing=False,
                        resampling=self.resampling,
                        **self.spec.vrt_params,
                    )
            else:
                logger.info(f"Skipping VRT for {self.url!r}")
                vrt = None

        if ds.driver in MULTITHREADED_DRIVER_ALLOWLIST:
            return ThreadLocalRioDataset(self.gdal_env, ds, vrt=vrt)
            # ^ NOTE: this forces all threads to wait for the `open()` we just did before they can open their
            # thread-local datasets. In principle, this would double the wall-clock open time, but if the above `open()`
            # is cached, it can actually be faster than all threads duplicating the same request in parallel.
            # This is worth profiling eventually for cases when STAC tells us the media type is a GeoTIFF.
        else:
            # logger.warning(
            #     f"Falling back on single-threaded reader for {self.url!r} (driver: {ds.driver!r}). "
            #     "This will be slow!"
            # )
            return SingleThreadedRioDataset(self.gdal_env, ds, vrt=vrt)


class CusotomSentinel1Reader:
    """Custom Sentinel 1 reader to avoid errors with stackstac"""

    def __init__(self, url, **kwargs):
        self.url = url

    def read(self, window: Window, **kwargs):
        print(f"read {self.url}")
        with rasterio.open(self.url) as src:
            print(f"reading {self.url}")
            result = src.read(1, masked=False, **kwargs)

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


def get_aoi_from_landsat_shub_catalog(
    shub_catalog: SentinelHubCatalog,
    data_collection: DataCollection,
    bbox: BBox,
    time_interval: Tuple[str, str],
    search_params: SearchParams,
    category: str = "T1",
) -> xr.DataArray:
    """Get an aoi from landsat shub catalog using the search params.
    If the aoi is not chunk aligned an offset will be added.

    Args:
        shub_catalog (SentinelHubCatalog): the catalog object, ex: catalog = SentinelHubCatalog(config=config)
        data_collection (DataCollection): the enum defining which landsat to use. ex: DataCollection.LANDSAT_ETM_L1
        bbox: (BBox): the bounding box. Ex: BBox((min_x, min_y, max_x, max_y), crs=CRS.WGS84)
        time_interval (Tuple[str, str]): format: ("2020-12-10", "2021-02-01")
        search_params (SearchParams): the search params to be used by pystac_client search.
        category (str): Tier level "RAW", "T1", "T2"

    Returns:
        xr.DataArray: the aoi as an xarray dataarray
    """

    search_iterator = shub_catalog.search(
        data_collection, bbox=bbox, time=time_interval, **search_params
    )
    processing_level = data_collection.processing_level
    items = []
    for res in search_iterator:
        try:
            if res["properties"]["landsat:collection_category"] == category:
                base_url = res["assets"]["data"]["href"]
                asset_id = base_url.split("/")[-1]
                if processing_level == "L1":
                    asset_id = asset_id.replace("_SR", "")
                mtl_url = f"{base_url}/{asset_id}_MTL.xml"
                item = landsat.create_stac_item(mtl_url)
                items.append(item)
        except FileNotFoundError:
            logger.warning(f"File {mtl_url} not found in s3 catalog.")

    # # Count the number of unique epsg in items
    # epsg_count = Counter([item.properties["proj:epsg"] for item in items])
    # # Select most common epsg
    # epsg = epsg_count.most_common(1)[0][0]
    # # Filter out items that doesn't have most common epsg
    # items = [item for item in items if item.properties["proj:epsg"] == epsg]
    items = pystac.ItemCollection(items)

    stack = stackstac.stack(items, epsg=4326)

    aoi = stack.loc[..., bbox.max_y : bbox.min_y, bbox.min_x : bbox.max_x]
    return aoi


def get_aoi_from_s1_shub_catalog(
    shub_catalog: SentinelHubCatalog,
    data_collection: DataCollection,
    bbox: BBox,
    time_interval: Tuple[str, str],
    search_params: SearchParams,
) -> xr.DataArray:
    """Get an aoi from sentinel 1 shub catalog using the search params.
    If the aoi is not chunk aligned an offset will be added.

    Args:
        shub_catalog (SentinelHubCatalog): the catalog object, ex: catalog = SentinelHubCatalog(config=config)
        data_collection (DataCollection): the enum defining which landsat to use. ex: DataCollection.SENTINEL1
        bbox: (BBox): the bounding box. Ex: BBox((min_x, min_y, max_x, max_y), crs=CRS.WGS84)
        time_interval (Tuple[str, str]): format: ("2020-12-10", "2021-02-01")
        search_params (SearchParams): the search params to be used by pystac_client search.

    Returns:
        xr.DataArray: the aoi as an xarray dataarray
    """

    search_iterator = shub_catalog.search(
        collection=data_collection, time=time_interval, bbox=bbox, **search_params
    )

    items = []
    for res in search_iterator:
        items.append(sentinel1.create_item(res["assets"]["s3"]["href"]))
    items = pystac.ItemCollection(items)
    stack = stackstac.stack(
        items, reader=AutoParallelRioReaderWithCrs, epsg=4326, fill_value=0
    )
    aoi = stack.loc[..., bbox.max_y : bbox.min_y, bbox.min_x : bbox.max_x]
    return aoi


def get_aoi_from_stac_catalog(
    catalog: Union[str, SentinelHubCatalog],
    data_collection: Union[str, DataCollection],
    bbox: BBox,
    time_interval: Tuple[str, str],
    search_params: SearchParams,
    **kwargs,
) -> xr.DataArray:
    """Get an aoi from a stac catalog using the search params.

    Args:
        catalog_url (Union[str, SentinelHubCatalog]): the catalog url
        data_collection (Union[str, DataCollection]): if using a SentinelHubCatalog url, it must be a DataCollection
        bbox (BBox): the bounding box. Ex: BBox((min_x, min_y, max_x, max_y), crs=CRS.WGS84)
        time_interval (Tuple[str, str]): format: ("2020-12-10", "2021-02-01")
        search_params (SearchParams): the search params to be used by the catalog search.

    Returns:
        xr.DataArray: the aoi as an xarray dataarray
    """
    collection_str = str(data_collection).lower()
    if "sentinel-s2" in collection_str:
        aoi = get_aoi_from_s2_stac_catalog(
            catalog, data_collection, bbox, time_interval, search_params, **kwargs
        )
    elif "sentinel1" in collection_str:
        aoi = get_aoi_from_s1_shub_catalog(
            catalog, data_collection, bbox, time_interval, search_params, **kwargs
        )
    elif "landsat" in collection_str:
        aoi = get_aoi_from_landsat_shub_catalog(
            catalog, data_collection, bbox, time_interval, search_params, **kwargs
        )
    else:
        raise Exception("sentinel2|sentinel1|landsat not found in data collection.")
    return aoi


def get_aoi_from_s2_stac_catalog(
    catalog: str,
    data_collection: str,
    bbox: BBox,
    time_interval: Tuple[str, str],
    search_params: SearchParams,
) -> xr.DataArray:
    """Get an aoi from a stac catalog using the search params.
    If the aoi is not chunk aligned an offset will be added.

    Args:
        catalog (str): the catalog
        data_collection (str): data collection in the stac catalog
        bbox (BBox): the bounding box. Ex: BBox((min_x, min_y, max_x, max_y), crs=CRS.WGS84)
        time_interval (Tuple[str,str]): time_interval. Format: ("2020-12-10","2021-02-01")
        search_params (SearchParams): the search params to be used by pystac_client search.

    Returns:
        xr.DataArray: the aoi as an xarray dataarray
    """
    catalog = pystac_client.Client.open(catalog)
    items = catalog.search(
        collections=data_collection,
        bbox=(bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y),
        datetime="/".join(time_interval),
        **search_params,
    ).get_all_items()
    print(items.items[0].properties)
    stack = stackstac.stack(items, epsg=4326)

    aoi = stack.loc[..., bbox.max_y : bbox.min_y, bbox.min_x : bbox.max_x]

    return aoi
