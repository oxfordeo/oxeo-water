import functools
from typing import List

import pyproj

from oxeo.core.constants import BAND_INFO


def identity(x):
    return x


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
