from attr import define


class Predictor:
    def predict(self):
        raise NotImplementedError


@define
class ModelDef:
    predictor: Predictor
