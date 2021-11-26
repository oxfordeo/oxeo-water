from typing import Optional

import numpy as np
from attr import define
from rasterio.fill import fillnodata
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


def pekel_bands(arr: np.ndarray, constellation) -> PekelBands:
    """Get all bands needed for Pekel masks."""

    assert arr.ndim == 3, "arr must have ndim == 3"

    cloud = cloud_band(constellation)
    bands = ["red", "green", "blue", "nir", "swir1", "swir2", "derived:ndvi", "alpha"]
    bands.append(cloud)

    # Scale sat-extractor bands to 0-1
    for i in range(6):

        a = np.clip(arr[..., i], 0, 10000)
        arr[..., i] = np.interp(a, (0, 10000), (0, 1))

    arr = fill_nodata_single(arr, alpha_band=bands.index("alpha"), erode=True)

    hsv = rgb2hsv(
        arr[..., [bands.index("swir2"), bands.index("nir"), bands.index("red")]]
    )
    hsv2 = rgb2hsv(
        arr[..., [bands.index("nir"), bands.index("green"), bands.index("blue")]]
    )

    b = PekelBands(
        alpha=arr[..., bands.index("alpha")],
        blue=arr[..., bands.index("blue")],
        swir1=arr[..., bands.index("swir1")],
        swir2=arr[..., bands.index("swir2")],
        hue=hsv[..., 0] * 360,
        sat=hsv[..., 1],
        val=hsv[..., 2],
        hue2=hsv2[..., 0] * 360,
        sat2=hsv2[..., 1],
        val2=hsv2[..., 2],
        cloud=arr[..., bands.index(cloud)],
        ndvi=arr[..., bands.index("derived:ndvi")],
    )
    if "sentinel" not in constellation:
        b.tirs1 = arr[..., bands.index("tirs1")] + 273.15

    return b


def fill_nodata_single(frame, alpha_band, erode=False):
    """Fillnodata on all bands UNTIL alpha band."""

    orig_dtype = frame.dtype
    frame = frame.copy().astype(np.float32)
    sel = np.where(frame[:, :, alpha_band] == 0)
    frame[:, :, :alpha_band][sel] = -1

    for b in range(alpha_band):
        band = frame[:, :, b].copy()
        mask = (band != -1).astype(np.int8)

        if erode:
            # Do an erosion to cover artifacts on the edge of nodata sections
            mask = erosion(mask, selem=np.ones((3, 3)))
        band = fillnodata(band, mask)
        frame[:, :, b] = band.copy()
    return frame.astype(orig_dtype)


def cloud_band(contellation):
    """Get the suitable cloud band for the given scene ID."""

    if "LT04" in contellation or "LT05" in contellation or "LE07" in contellation:
        cloud = "derived:visual_cloud_mask"
    elif "LC08" in contellation or "sentinel-2" in contellation:
        cloud = "cloud-mask"
    else:
        raise NotImplementedError(f"No cloud band set for for {contellation}")
    return cloud
