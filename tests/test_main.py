import numpy as np

from oxeo.water.main import run_model


def test_run_model():
    stack = np.array([1, 2, 3])
    water = run_model(stack)
    assert (water == np.array([False, True, True])).all()
