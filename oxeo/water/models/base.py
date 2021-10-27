from attr import define, frozen


@frozen
class Predictor:
    def predict(self):
        raise NotImplementedError


@define
class ModelDef:
    predictor: Predictor
