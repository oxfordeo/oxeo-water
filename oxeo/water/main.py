import numpy as np

from oxeo.water.models.factory import model_factory


def run_model(stack: np.ndarray):
    return model_factory("pekel").predictor().predict(stack)
