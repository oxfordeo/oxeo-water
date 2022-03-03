from typing import Any, List

import attrs
from attrs import define

from oxeo.core.models.timeseries import TimeseriesMask, merge_masks_all_constellations
from oxeo.core.models.waterbody import WaterBody
from oxeo.water.models.base import Predictor
from oxeo.water.models.factory import model_factory
from oxeo.water.models.tile_utils import predict_tile


@define
class WaterBodyPredictor:
    fs: Any
    model_name: str
    revisit_chunk_size: int
    ckpt_path: str = attrs.field(default=None)
    batch_size: int = attrs.field(default=None)
    bands: list[str] = attrs.field(default=None)
    target_size: int = attrs.field(default=None)
    predictor: Predictor = attrs.field(init=False)

    def __attrs_post_init__(self):
        self.predictor = model_factory(self.model_name).predictor(
            ckpt_path=self.ckpt_path,
            fs=self.fs,
            batch_size=self.batch_size,
            bands=self.bands,
            target_size=self.target_size,
        )

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
                self.predictor,
                self.revisit_chunk_size,
                start_date,
                end_date,
                fs,
                overwrite=False,
                gpu=gpu,
            )

        return merge_masks_all_constellations(waterbody, self.model_name)
