from oxeo.water.models.base import ModelDef
from oxeo.water.models.cloud_mask import CloudMaskPredictor
from oxeo.water.models.pekel import PekelPredictor
from oxeo.water.models.segmentation import Segmentation2DPredictor


def model_factory(name: str) -> ModelDef:
    model_list = {
        "pekel": ModelDef(
            predictor=PekelPredictor,
        ),
        "cloud_mask": ModelDef(
            predictor=CloudMaskPredictor,
        ),
        "cnn": ModelDef(
            predictor=Segmentation2DPredictor,
        ),
    }
    return model_list[name]
