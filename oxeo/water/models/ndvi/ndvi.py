from typing import Optional, Tuple, Union

from sentinelhub import BBox, DataCollection, SentinelHubCatalog
from stackstac.rio_env import LayeredEnv

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
        resolution: int = 10,
        env: Optional[LayeredEnv] = None,
        **kwargs,
    ):
        aoi = get_aoi_from_stac_catalog(
            catalog=catalog,
            data_collection=data_collection,
            bbox=bbox,
            time_interval=time_interval,
            search_params=search_params,
            resolution=resolution,
            env=env,
            **kwargs,
        )
        bands = aoi.band.values
        red_band_index = [
            i for i, b in enumerate(bands) if b and ("red" in b or "B04" in b)
        ][0]
        nir_band_index = [
            i for i, b in enumerate(bands) if b and ("nir" in b or "B08" in b)
        ][0]

        nir_band = aoi.isel(band=nir_band_index)
        red_band = aoi.isel(band=red_band_index)
        ndvi = (nir_band - red_band) / (nir_band + red_band)
        return ndvi
