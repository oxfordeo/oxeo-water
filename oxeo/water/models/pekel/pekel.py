import numpy as np
from attr import frozen

from oxeo.water.models import Predictor


@frozen
class PekelPredictor(Predictor):
    def predict(self, arr: np.ndarray):
        return arr > 1
