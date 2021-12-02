from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from oxeo.water.models.utils import TimeseriesMask

UNITS = ["pixel", "meter"]


def segmentation_area_multiple(segs: List[TimeseriesMask]) -> pd.DataFrame:
    dfs = []
    for tsm in segs:
        area = segmentation_area(tsm.mask, unit="meter", resolution=tsm.resolution)
        df = pd.DataFrame(
            data={
                "date": tsm.dates,
                "area": area,
                "constellation": tsm.constellation,
            }
        )
        dfs.append(df)
    return pd.concat(dfs, axis=0)


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
        total_area *= resolution ** 2

    return total_area


def pearson(x, y):
    return stats.pearsonr(x, y)[0]
