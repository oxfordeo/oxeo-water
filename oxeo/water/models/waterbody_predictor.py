from typing import Any, Dict, List

from attrs import define

from oxeo.core.models.timeseries import TimeseriesMask, merge_masks_all_constellations
from oxeo.core.models.waterbody import WaterBody
from oxeo.water.models.tile_utils import predict_tile


@define
class WaterBodyPredictor:
    fs: Any
    model_name: str
    predictor_kwargs: Dict[str, Any]
    revisit_chunk_size: int

    def predict(
        self,
        waterbody: WaterBody,
        start_date: str,
        end_date: str,
        fs: Any = None,
        gpu=0,
    ) -> List[TimeseriesMask]:
        tile_paths = waterbody.paths
        for t_path in tile_paths:

            predict_tile(
                t_path,
                self.model_name,
                self.predictor_kwargs,
                self.revisit_chunk_size,
                start_date,
                end_date,
                fs,
                overwrite=False,
                gpu=gpu,
            )

        return merge_masks_all_constellations(waterbody, self.model_name)
