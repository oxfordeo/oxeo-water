from collections import defaultdict
from typing import Callable, Dict, List

import numpy as np
import zarr
from scipy import stats
from tqdm import tqdm
from zarr.errors import PathNotFoundError

from oxeo.core.logging import logger
from oxeo.core.models.tile import TilePath, load_tile_as_dict, tile_from_id


def precision(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """Return the precision of two arrays

    Args:
        arr1 (np.ndarray): The first array
        arr2 (np.ndarray): The second array
    """
    assert arr1.shape == arr2.shape, "Arrays must have the same shape"
    assert arr1.dtype == arr2.dtype, "Arrays must have the same type"

    return np.sum(arr1 * arr2) / np.sum(arr1)


def recall(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """Return the recall of two arrays

    Args:
        arr1 (np.ndarray): The first array
        arr2 (np.ndarray): The second array
    """
    assert arr1.shape == arr2.shape, "Arrays must have the same shape"
    assert arr1.dtype == arr2.dtype, "Arrays must have the same type"

    return np.sum(arr1 * arr2) / np.sum(arr2)


def pearson(x, y):
    return stats.pearsonr(x, y)[0]


def accuracy(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """Return the accuracy of two arrays

    Args:
        arr1 (np.ndarray): The first array
        arr2 (np.ndarray): The second array
    """
    assert arr1.shape == arr2.shape, "Arrays must have the same shape"
    assert arr1.dtype == arr2.dtype, "Arrays must have the same type"

    return np.sum(arr1 == arr2) / arr1.size


def iou(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """Return the intersection over union of two arrays

    Args:
        arr1 (np.ndarray): The first array
        arr2 (np.ndarray): The second array
    """
    assert arr1.shape == arr2.shape, "Arrays must have the same shape"
    assert arr1.dtype == arr2.dtype, "Arrays must have the same type"

    intersection = np.sum(arr1 * arr2)
    union = np.sum(arr1) + np.sum(arr2) - intersection
    return intersection / union


def dice(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """Return the dice coefficient of two arrays

    Args:
        arr1 (np.ndarray): The first array
        arr2 (np.ndarray): The second array
    """
    assert arr1.shape == arr2.shape, "Arrays must have the same shape"
    assert arr1.dtype == arr2.dtype, "Arrays must have the same type"

    intersection = np.sum(arr1 * arr2)
    return 2 * intersection / (np.sum(arr1) + np.sum(arr2))


def multiclass_metric(
    metric: Callable, arr1: np.ndarray, arr2: np.ndarray
) -> Dict[int, float]:
    """Return an dict containing the metric for each class in arr1 and arr2

    Args:
        arr1 (np.ndarray): The first array
        arr2 (np.ndarray): The second array
    """
    assert arr1.shape == arr2.shape, "Arrays must have the same shape"
    assert arr1.dtype == arr2.dtype, "Arrays must have the same type"

    arr1_classes = np.unique(arr1)
    arr2_classes = np.unique(arr2)
    arr_classes = np.union1d(arr1_classes, arr2_classes)

    metric_dict = {}
    for c in arr_classes:
        metric_dict[c] = metric(arr1 == c, arr2 == c)

    return metric_dict


def tile_stat_per_band(
    fs_mapper,
    tile_ids: List[str],
    bands: List[str],
    constellations: List[str],
    revisits_batch: int = 32,
    stat: Callable = None,
) -> Dict[str, Dict[str, float]]:
    """Calculate the tiles mean per band for every listed constellation

    Args:
        tile_ids (List[str]): The tile ids to include in the mean
        bands (List[str]): The bands to calculate the mean
        constellations (List[str]): The constellations.
        revisits_batch (int): how many revisits to load per step when loading tile.
                                reduce to avoid memory consumption.
        stat (Callable): function to run on tile. Should be a np callable (ie np.mean)

    Returns:
        Dict[str, Dict[str, float]]: A dictionary with each constellation as key, and
                    a dictionary with band_name:value as value
    """
    res = defaultdict(list)
    for tile_id in tqdm(tile_ids):
        for constellation in constellations:
            tile = tile_from_id(tile_id)
            tile_path = TilePath(tile, constellation)
            try:
                zarr_arr = zarr.open(tile_path.data_path, "r")
                revisits = zarr_arr.shape[0]
                for i in range(0, revisits, revisits_batch):
                    tile_tensor = load_tile_as_dict(
                        fs_mapper=fs_mapper,
                        tile_path=tile_path,
                        revisit=slice(i, i + revisits_batch),
                        bands=bands,
                    )
                    tile_mean = stat(tile_tensor["image"], axis=(2, 3))
                    res[constellation].extend(tile_mean.tolist())
            except PathNotFoundError:
                logger.error("Path {tile_path.data_path} not found.")

    for key in res.keys():
        res[key] = np.array(res[key]).mean(0)

    return {
        c: {bands[i]: values[i] for i, _ in enumerate(values)}
        for c, values in res.items()
    }
