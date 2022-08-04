import math
from typing import Tuple, Union

import dask.array as da
from sentinelhub import BBox, DataCollection, SentinelHubCatalog

from oxeo.core.data import SearchParams, get_aoi_from_stac_catalog
from oxeo.water.models.base import Predictor


class SoilMoisturePredictor(Predictor):
    def DpRVIc(self, aoi):
        vv = aoi.sel(band="vv")
        vh = aoi.sel(band="vh")

        # Paper code to calculate soil moisture:

        C11_mean = vv.rolling(x=5, y=5, center=True).mean()
        C22_mean = vh.rolling(x=5, y=5, center=True).mean()

        span = C11_mean + C22_mean
        ratio = C22_mean / C11_mean
        vmask = C11_mean - C22_mean
        vmask = vmask > 0

        m = abs(C11_mean - C22_mean) / span

        C11_mean - C22_mean
        theta_c = da.arctan(
            (abs(C11_mean - C22_mean) * span * m)
            / (C11_mean * C22_mean + (da.power(span, 2) * da.power(m, 2)))
        )

        theta_c = theta_c * 180 / math.pi

        q = ratio

        DpRVIc_n = q * (q + 3)
        DpRVIc_d = (q + 1) * (q + 1)

        DpRVIc = DpRVIc_n / DpRVIc_d

        C11_mean_db = da.log10(C11_mean) * 10
        C11_rc = C11_mean_db >= -17

        DpRVIc = DpRVIc * vmask * C11_rc
        return DpRVIc

    def predict_stac_aoi(
        self,
        catalog: Union[str, SentinelHubCatalog],
        data_collection: Union[str, DataCollection],
        bbox: BBox,
        time_interval: Tuple[str, str],
        search_params: SearchParams,
        resolution: int = 10,
    ):
        aoi = get_aoi_from_stac_catalog(
            catalog=catalog,
            data_collection=data_collection,
            bbox=bbox,
            time_interval=time_interval,
            search_params=search_params,
            orbit_state="descending",
            resolution=resolution,
        )

        DpRVIc = self.DpRVIc(aoi)
        return DpRVIc
