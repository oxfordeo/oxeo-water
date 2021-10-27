from oxeo.water.models import ModelDef
from oxeo.water.models.pekel import PekelPredictor


def model_factory(name: str):
    model_list = {
        "pekel": ModelDef(
            predictor=PekelPredictor,
        ),
    }
    return model_list[name]
