from typing import Tuple, Union

from sentinelhub import BBox, DataCollection, SentinelHubCatalog

from oxeo.core.data import SearchParams, get_aoi_from_stac_catalog
from oxeo.water.models.base import Predictor


class NDVIPredictor(Predictor):
    def predict_stac_aoi(
        self,
        catalog: Union[str, SentinelHubCatalog],
        data_collection: Union[str, DataCollection],
        bbox: BBox,
        time_interval: Tuple[str, str],
        search_params: SearchParams,
        **kwargs,
    ):
        aoi = get_aoi_from_stac_catalog(
            catalog=catalog,
            data_collection=data_collection,
            bbox=bbox,
            time_interval=time_interval,
            search_params=search_params,
            **kwargs,
        )
        common_names = aoi.common_name.values
        red_band_index = [i for i, b in enumerate(common_names) if b and "red" in b][0]
        nir_band_index = [i for i, b in enumerate(common_names) if b and "nir" in b][0]

        nir_band = aoi.isel(band=nir_band_index)
        red_band = aoi.isel(band=red_band_index)
        ndvi = (nir_band - red_band) / (nir_band + red_band)
        return ndvi
