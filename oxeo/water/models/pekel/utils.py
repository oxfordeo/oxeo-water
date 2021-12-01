from typing import List, Optional

import numpy as np
from attr import define
from rasterio.fill import fillnodata
from satextractor.models.constellation_info import BAND_INFO
from skimage.color import rgb2hsv
from skimage.morphology import erosion


@define
class PekelBands:
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
    return [b["band"].common_name for b in BAND_INFO[constellation].values()]


def pekel_bands(arr: np.ndarray, constellation: str) -> PekelBands:
    """Get all bands needed for Pekel masks."""

    assert arr.ndim == 3, "arr must have ndim == 3"

    bands = get_band_list(constellation)

    ndvi = (arr[bands.index("nir")] - arr[bands.index("red")]) / (
        arr[bands.index("nir")] + arr[bands.index("red")]
    )
    alpha = np.ones(arr.shape[1:])
    cloud_mask = np.zeros(arr.shape[1:])

    # Scale sat-extractor bands to 0-1
    for i in range(6):
        a = np.clip(arr[i, ...], 0, 10000)
        arr[i, ...] = np.interp(a, (0, 10000), (0, 1))

    # Don't fill_nodata, as we don't have a real alpha band!
    # arr = fill_nodata_single(arr, alpha, erode=True)

    hsv = rgb2hsv(
        arr[
            [bands.index("swir2"), bands.index("nir"), bands.index("red")], ...
        ].transpose(1, 2, 0)
    )
    hsv2 = rgb2hsv(
        arr[
            [bands.index("nir"), bands.index("green"), bands.index("blue")], ...
        ].transpose(1, 2, 0)
    )

    b = PekelBands(
        alpha=alpha,
        blue=arr[bands.index("blue"), ...],
        swir1=arr[bands.index("swir1"), ...],
        swir2=arr[bands.index("swir2"), ...],
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


def fill_nodata_single(arr, alpha, erode=False):
    """Fillnodata on all bands."""

    orig_dtype = arr.dtype
    frame = arr.copy().astype(np.float32)
    sel = np.where(alpha == 0)
    frame[sel] = -1

    for b in range(frame.shape[0]):
        band = frame[b, ...].copy()
        mask = (band != -1).astype(np.int8)

        if erode:
            # Do an erosion to cover artifacts on the edge of nodata sections
            mask = erosion(mask, selem=np.ones((3, 3)))
        band = fillnodata(band, mask)
        frame[b, ...] = band.copy()
    return frame.astype(orig_dtype)
