from typing import Any

import numpy as np
from attr import frozen
from joblib import Parallel, delayed
from satextractor.utils import tqdm_joblib
from tqdm import tqdm

from oxeo.water.models import Predictor
from oxeo.water.models.pekel import masks, utils


@frozen
class PekelPredictor(Predictor):
    def predict_single_revisit(
        self,
        arr: Any,
        revisit: int,
        bands_indexes: dict,
        constellation="sentinel-2",
        compute=False,
    ):
        """Return mask for a BxHxW array
        bands_indexes are the indexes for these bands:
        ["red", "green", "blue", "nir", "swir1", "swir2"]

        Example: {
            "red": 3,
            "green": 2,
            "blue": 1,
            "nir": 7,
            "swir1": 11,
            "swir2": 12,
        },

        Args:
            arr (np.ndarray): the array to make predictions in

        Returns:
            [type]: HxW mask
        """
        arr = arr[revisit]
        arr = arr[
            [
                bands_indexes["red"],
                bands_indexes["green"],
                bands_indexes["blue"],
                bands_indexes["nir"],
                bands_indexes["swir1"],
                bands_indexes["swir2"],
            ],
        ]
        if compute:
            arr = arr.compute()
        arr = arr.astype(float)

        # ["red", "green", "blue", "nir", "swir1", "swir2", "derived:ndvi", "alpha", "cloud_mask"]

        ndvi = np.expand_dims((arr[3] - arr[0]) / (arr[3] + arr[0]), axis=0)

        alpha = np.ones((1, *arr.shape[1:]))
        cloud_mask = np.zeros((1, *arr.shape[1:]))

        arr = np.append(arr, ndvi, axis=0)
        arr = np.append(arr, alpha, axis=0)
        arr = np.append(arr, cloud_mask, axis=0)

        p_bands = utils.pekel_bands(arr.transpose(1, 2, 0), constellation)
        c_masks = masks.combine_masks(p_bands, False)
        return c_masks

    def predict(
        self,
        arr: Any,
        bands_indexes: dict,
        constellation="sentinel-2",
        compute=False,
        n_jobs=-1,
        verbose=0,
    ):
        """Return masks for a TxBxHxW array
        bands_indexes are the indexes for these bands:
        ["red", "green", "blue", "nir", "swir1", "swir2"]

        Example: {
            "red": 3,
            "green": 2,
            "blue": 1,
            "nir": 7,
            "swir1": 11,
            "swir2": 12,
        },

        Args:
            arr (np.ndarray): the array to make predictions in

        Returns:
            [type]: TxHxW masks
        """
        with tqdm_joblib(
            tqdm(
                desc="parallel predicting masks on revistis.",
                total=arr.shape[0],
            ),
        ):
            masks = Parallel(n_jobs=n_jobs, verbose=verbose)(
                [
                    delayed(self.predict_single_revisit)(
                        arr, revisit, bands_indexes, constellation, compute
                    )
                    for revisit in range(arr.shape[0])
                ],
            )

        return masks
