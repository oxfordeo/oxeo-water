from typing import Any, Dict, Iterable

import numpy as np
import torch
from torch import Tensor


def np_index(array, item):
    for idx, val in np.ndenumerate(array):
        if val == item:
            return idx


def merge_samples(samples: Iterable[Dict[Any, Any]]) -> Dict[Any, Any]:
    """Merge a list of samples.

    Useful for joining samples in a :class:`datasets.UnionDataset`.

    Args:
        samples: list of samples

    Returns:
        a single sample
    """
    collated: Dict[Any, Any] = {}
    for sample in samples:
        for key, value in sample.items():
            if key in collated and isinstance(value, Tensor):
                # Take the maximum so that nodata values (zeros) get replaced
                # by data values whenever possible
                collated[key] = torch.maximum(  # type: ignore[attr-defined]
                    collated[key], value
                )
            else:
                collated[key] = value
    return collated
