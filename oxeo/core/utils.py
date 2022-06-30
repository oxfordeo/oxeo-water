import functools
from typing import List

import geojson
import numpy as np
import pyproj

from oxeo.core.constants import BAND_INFO


def identity(x):
    return x


def get_bounding_box(geometry):
    coords = np.array(list(geojson.utils.coords(geometry)))
    return (
        coords[:, 0].min(),
        coords[:, 1].min(),
        coords[:, 0].max(),
        coords[:, 1].max(),
    )


@functools.lru_cache(maxsize=5)
def get_transform_function(crs_from: str, crs_to: str, always_xy=True):
    return pyproj.Transformer.from_proj(
        projection(crs_from),
        projection(crs_to),
        always_xy=always_xy,
    ).transform


@functools.lru_cache(maxsize=5)
def projection(crs):
    if crs == "WGS84":
        proj_str = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
    else:
        proj_str = f"EPSG:{crs}"
    return pyproj.Proj(proj_str, preserve_units=True)


def get_band_list(constellation: str) -> List[str]:
    return [b["band"].common_name for b in BAND_INFO[constellation].values()]


def get_utm_zone(lat, lon):
    """A function to grab the UTM zone number for any lat/lon location"""
    zone_str = str(int((lon + 180) / 6) + 1)

    if (lat >= 56.0) & (lat < 64.0) & (lon >= 3.0) & (lon < 12.0):
        zone_str = "32"
    elif (lat >= 72.0) & (lat < 84.0):
        if (lon >= 0.0) & (lon < 9.0):
            zone_str = "31"
        elif (lon >= 9.0) & (lon < 21.0):
            zone_str = "33"
        elif (lon >= 21.0) & (lon < 33.0):
            zone_str = "35"
        elif (lon >= 33.0) & (lon < 42.0):
            zone_str = "37"

    return zone_str


def get_utm_epsg(lat, lon, utm_zone=None):
    """A function to combine the UTM zone number and the hemisphere into an EPSG code"""

    if utm_zone is None:
        utm_zone = get_utm_zone(lat, lon)

    if lat > 0:
        return int(f"{str(326)+str(utm_zone)}")
    else:
        return int(f"{str(327)+str(utm_zone)}")
