import numpy as np
from satextractor.models.constellation_info import BAND_INFO

from .mtl_parser import parsemeta


def to_toa_radiance(arr: np.ndarray, mtl_dict: dict, constellation: str):

    bands = list(BAND_INFO[constellation].keys())
    arr = arr.astype(float)

    radiometric_rescaling = mtl_dict["L1_METADATA_FILE"]["RADIOMETRIC_RESCALING"]

    for i, b in enumerate(bands):
        radiance_add_band = f"RADIANCE_ADD_BAND_{b.replace('B','')}"
        radiance_mult_band = f"RADIANCE_MULT_BAND_{b.replace('B','')}"

        radiance_add = radiometric_rescaling.get(radiance_add_band)
        radiance_mult = radiometric_rescaling.get(radiance_mult_band)

        if radiance_add is not None and radiance_mult is not None:
            arr[i] = radiance_mult * arr[i] + radiance_add
    return arr


def to_toa_reflectance(arr: np.ndarray, mtl_dict: dict, constellation: str):
    bands = list(BAND_INFO[constellation].keys())
    arr = arr.astype(float)

    radiometric_rescaling = mtl_dict["L1_METADATA_FILE"]["RADIOMETRIC_RESCALING"]

    sun_elevation = mtl_dict["L1_METADATA_FILE"]["IMAGE_ATTRIBUTES"]["SUN_ELEVATION"]
    for i, b in enumerate(bands):
        reflectance_add_band = f"REFLECTANCE_ADD_BAND_{b.replace('B','')}"
        reflectance_mult_band = f"REFLECTANCE_MULT_BAND_{b.replace('B','')}"

        reflectance_add = radiometric_rescaling.get(reflectance_add_band)
        reflectance_mult = radiometric_rescaling.get(reflectance_mult_band)

        if reflectance_add is not None and reflectance_mult is not None:
            arr[i] = (reflectance_mult * arr[i] + reflectance_add) / np.sin(
                sun_elevation
            )
    return arr


def to_toa_brightness_temp(arr: np.ndarray, mtl_dict: dict, constellation: str):
    bands = list(BAND_INFO[constellation].keys())
    arr = arr.astype(float)

    radiometric_rescaling = mtl_dict["L1_METADATA_FILE"]["RADIOMETRIC_RESCALING"]

    for i, b in enumerate(bands):
        radiance_add_band = f"RADIANCE_ADD_BAND_{b.replace('B','')}"
        radiance_mult_band = f"RADIANCE_MULT_BAND_{b.replace('B','')}"

        k1_constant_band = f"K1_CONSTANT_BAND_{b.replace('B','')}"
        k2_constant_band = f"K1_CONSTANT_BAND_{b.replace('B','')}"

        k1 = radiometric_rescaling.get(k1_constant_band)
        k2 = radiometric_rescaling.get(k2_constant_band)

        radiance_add = radiometric_rescaling.get(radiance_add_band)
        radiance_mult = radiometric_rescaling.get(radiance_mult_band)

        if k1 is not None and k2 is not None:

            rad = radiance_mult * arr[i] + radiance_add

            arr[i] = k2 / np.log(k1 / rad + 1)

    return arr


def to_toa(arr: np.ndarray, mtl_dict: dict, constellation: str):
    ref = to_toa_reflectance(arr, mtl_dict, constellation)
    return to_toa_brightness_temp(ref, mtl_dict, constellation)


def get_mtl_dict(fs, mtl_path: str, revisit: int):
    f = fs.open([path for path in fs.ls(mtl_path)][revisit])
    return parsemeta(f.read().decode("utf-8"))
