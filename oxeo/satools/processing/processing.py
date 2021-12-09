from typing import List

import attr
import numpy as np
import xarray as xr


@attr.s
class BandTOAValues:
    band = attr.ib()
    radiance_mult = attr.ib(default=None)
    radiance_add = attr.ib(default=None)
    reflectance_mult = attr.ib(default=None)
    reflectance_add = attr.ib(default=None)
    k1_constant = attr.ib(default=None)
    k2_constant = attr.ib(default=None)


def to_toa_reflectance(arr: xr.DataArray, band_toa_values: List[BandTOAValues]):
    arr = arr.astype(float)
    for b_toa in band_toa_values:
        arr.loc[dict(bands=b_toa.band)] = (
            b_toa.reflectance_mult * arr.loc[dict(bands=b_toa.band)]
            + b_toa.reflectance_add
        )
    return arr


def toa_brightness_temp(arr: xr.DataArray, band_toa_values: List[BandTOAValues]):
    arr = arr.astype(float)

    for b_toa in band_toa_values:
        current_band = arr.loc[dict(bands=b_toa.band)]
        toa_radiance = b_toa.radiance_mult * current_band + b_toa.radiance_add

        arr.loc[dict(bands=b_toa.band)] = b_toa.k2_constant / np.log(
            b_toa.k1_constant / toa_radiance + 1
        )

    return arr
