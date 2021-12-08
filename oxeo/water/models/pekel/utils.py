from typing import List, Optional

import numpy as np
from attr import define
from pystac.extensions.eo import Band
from satextractor.models.constellation_info import BAND_INFO
from skimage.color import rgb2hsv


@define
class Bands:
    pass


@define
class SarBands(Bands):
    vv: np.ndarray
    vh: np.ndarray


@define
class PekelBands(Bands):
    alpha: np.ndarray
    blue: np.ndarray
    swir1: np.ndarray
    swir2: np.ndarray
    hue: np.ndarray
    sat: np.ndarray
    val: np.ndarray
    hue2: np.ndarray
    sat2: np.ndarray
    val2: np.ndarray
    ndvi: np.ndarray
    cloud: np.ndarray
    tirs1: Optional[np.ndarray] = None


def get_band_list(constellation: str) -> List[str]:
    BAND_INFO["sentinel-1"] = {
        "B1": {"band": Band.create(name="B1", common_name="VV")},
        "B2": {"band": Band.create(name="B2", common_name="VH")},
    }
    return [b["band"].common_name for b in BAND_INFO[constellation].values()]


def pekel_bands(arr: np.ndarray, constellation: str) -> Bands:
    """Get all bands needed for Pekel masks."""

    assert arr.ndim == 3, "arr must have ndim == 3"

    bands = get_band_list(constellation)

    if constellation == "sentinel-1":
        vv = 2 ** 16 - arr[bands.index("VV")]
        vh = 2 ** 16 - arr[bands.index("VH")]
        return SarBands(vv=vv, vh=vh)

    arr = np.interp(arr, (0, 10000), (0, 1))

    alpha = np.ones(arr.shape[1:])
    cloud_mask = np.zeros(arr.shape[1:])

    nir = arr[bands.index("nir")]
    red = arr[bands.index("red")]
    green = arr[bands.index("green")]
    blue = arr[bands.index("blue")]
    swir1 = arr[bands.index("swir1")]
    swir2 = arr[bands.index("swir2")]

    ndvi = (nir - red) / (nir + red)
    hsv = rgb2hsv(np.stack((swir2, nir, red), axis=-1))
    hsv2 = rgb2hsv(np.stack((nir, green, blue), axis=-1))

    b = PekelBands(
        alpha=alpha,
        blue=blue,
        swir1=swir1,
        swir2=swir2,
        hue=hsv[..., 0] * 360,
        sat=hsv[..., 1],
        val=hsv[..., 2],
        hue2=hsv2[..., 0] * 360,
        sat2=hsv2[..., 1],
        val2=hsv2[..., 2],
        cloud=cloud_mask,
        ndvi=ndvi,
        tirs1=arr[bands.index("tirs1")] if "tirs1" in bands else None,
    )

    return b
