# from oxeo.water.models.cloud_mask import CloudMaskPredictor
# from oxeo.water.models.pekel import PekelPredictor
# from oxeo.water.models.segmentation import Segmentation2DPredictor
import importlib

from oxeo.water.models.base import ModelDef


def model_factory(name: str) -> ModelDef:
    model_list = {
        "pekel": ModelDef(
            predictor=importlib.import_module("oxeo.water.models.pekel").PekelPredictor,
        ),
        "cloud_mask": ModelDef(
            predictor=importlib.import_module(
                "oxeo.water.models.cloud_mask"
            ).CloudMaskPredictor,
        ),
        "cnn": ModelDef(
            predictor=importlib.import_module(
                "oxeo.water.models.segmentation"
            ).Segmentation2DPredictor,
        ),
    }
    return model_list[name]
