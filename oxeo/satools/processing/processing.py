import numpy as np
import xarray as xr

from .mtl_parser import parsemeta


def to_toa_radiance(arr: xr.DataArray, mtl_dict: dict):
    arr = arr.astype(float)

    radiometric_rescaling = mtl_dict["L1_METADATA_FILE"]["RADIOMETRIC_RESCALING"]

    for b in arr.bands.values:
        radiance_add_band = f"RADIANCE_ADD_BAND_{b.replace('B','')}"
        radiance_mult_band = f"RADIANCE_MULT_BAND_{b.replace('B','')}"

        radiance_add = radiometric_rescaling.get(radiance_add_band)
        radiance_mult = radiometric_rescaling.get(radiance_mult_band)

        if radiance_add is not None and radiance_mult is not None:
            arr.loc[dict(bands=b)] = (
                radiance_mult * arr.loc[dict(bands=b)] + radiance_add
            )
    return arr


def to_toa_reflectance(arr: xr.DataArray, mtl_dict: dict):
    arr = arr.astype(float)

    radiometric_rescaling = mtl_dict["L1_METADATA_FILE"]["RADIOMETRIC_RESCALING"]

    sun_elevation = mtl_dict["L1_METADATA_FILE"]["IMAGE_ATTRIBUTES"]["SUN_ELEVATION"]
    for b in arr.bands.values:
        reflectance_add_band = f"REFLECTANCE_ADD_BAND_{b.replace('B','')}"
        reflectance_mult_band = f"REFLECTANCE_MULT_BAND_{b.replace('B','')}"

        reflectance_add = radiometric_rescaling.get(reflectance_add_band)
        reflectance_mult = radiometric_rescaling.get(reflectance_mult_band)

        if reflectance_add is not None and reflectance_mult is not None:
            arr.loc[dict(bands=b)] = (
                reflectance_mult * arr.loc[dict(bands=b)] + reflectance_add
            ) / np.sin(sun_elevation)
    return arr


def to_toa_brightness_temp(arr: xr.DataArray, mtl_dict: dict):
    arr = arr.astype(float)

    radiometric_rescaling = mtl_dict["L1_METADATA_FILE"]["RADIOMETRIC_RESCALING"]

    for b in arr.bands.values:
        radiance_add_band = f"RADIANCE_ADD_BAND_{b.replace('B','')}"
        radiance_mult_band = f"RADIANCE_MULT_BAND_{b.replace('B','')}"

        k1_constant_band = f"K1_CONSTANT_BAND_{b.replace('B','')}"
        k2_constant_band = f"K1_CONSTANT_BAND_{b.replace('B','')}"

        k1 = radiometric_rescaling.get(k1_constant_band)
        k2 = radiometric_rescaling.get(k2_constant_band)

        radiance_add = radiometric_rescaling.get(radiance_add_band)
        radiance_mult = radiometric_rescaling.get(radiance_mult_band)

        if k1 is not None and k2 is not None:

            rad = radiance_mult * arr.loc[dict(bands=b)] + radiance_add

            arr.loc[dict(bands=b)] = k2 / np.log(k1 / rad + 1)

    return arr


def to_toa(arr: xr.DataArray, mtl_dict: dict):
    ref = to_toa_reflectance(arr, mtl_dict)
    return to_toa_brightness_temp(ref, mtl_dict)


def get_mtl_dict(fs, ds, position):
    revist_ds = ds.isel(dict(revisits=position))
    str_date = str(revist_ds.revisits.values)[:10]

    data_var = list(revist_ds.data_vars.keys())[0]
    mtl_path = f"{revist_ds.attrs['patch_files'][0]}/{data_var}/metadata"
    f = fs.open([path for path in fs.ls(mtl_path) if str_date in path][0])
    return parsemeta(f.read().decode("utf-8"))
