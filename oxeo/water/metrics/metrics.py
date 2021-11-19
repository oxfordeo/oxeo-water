from typing import Optional

import numpy as np
from scipy import stats

UNITS = ["pixel", "meter"]


def segmentation_area(
    seg: np.ndarray,
    unit: str = "pixel",
    resolution: Optional[int] = 1,
) -> np.ndarray:
    """Get the total area of a segmentation (Nx..xHxW)

    Args:
        seg (np.ndarray): N dimensional binary segmentation.
        unit (str): the unit of the area. Can be in pixels or meters.
        resolution (Optional[int]): if unit is meters the seg resolution must be present

    Returns:
        float: total area (Nx...)
    """
    assert unit in UNITS, f"unit must be one of {UNITS}"
    total_area = (seg > 0).sum(axis=(-2, -1))

    if unit == "meter":
        assert resolution is not None, "resolution is mandatory when unit is 'meter'"
        total_area *= resolution

    return total_area


def pearson(x, y):
    return stats.pearsonr(x, y)[0]
