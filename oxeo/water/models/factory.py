from oxeo.water.models import ModelDef
from oxeo.water.models.cloud_mask import CloudMaskPredictor
from oxeo.water.models.pekel import PekelPredictor


def model_factory(name: str) -> ModelDef:
    model_list = {
        "pekel": ModelDef(
            predictor=PekelPredictor,
        ),
        "cloud_mask": ModelDef(
            predictor=CloudMaskPredictor,
        ),
    }
    return model_list[name]
