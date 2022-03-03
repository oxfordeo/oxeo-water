from typing import Any

import numpy as np
import zarr
from attr import define
from joblib import Parallel, delayed
from pymasker import LandsatConfidence, LandsatMasker
from s2cloudless import S2PixelCloudDetector
from satextractor.models.constellation_info import BAND_INFO
from tqdm import tqdm

from oxeo.water.models.base import Predictor
from oxeo.water.utils import tqdm_joblib


@define
class CloudMaskPredictor(Predictor):
    def predict_single_revisit(
        self,
        arr: zarr.core.Array,
        revisit: int,
        constellation="sentinel-2",
    ):
        """Return mask for a BxHxW array
        Args:
            arr (zarr.core.Array): the zarr array to make predictions in

        Returns:
            [type]: HxW mask
        """
        arr.store.fs

        if "landsat" in constellation:

            bqa_index = list(BAND_INFO[constellation].keys()).index("BQA")
            masker = LandsatMasker(arr[revisit, bqa_index], collection=1)

            conf = LandsatConfidence.high
            cloud_mask = masker.get_cloud_mask(conf)
        elif constellation == "sentinel-2":
            sen2_cloud_detector = S2PixelCloudDetector(
                threshold=0.5, average_over=4, dilation_size=2, all_bands=True
            )
            arr = arr[revisit].transpose(1, 2, 0)
            arr = np.interp(arr, (0, 10000), (0, 1))
            cloud_mask = sen2_cloud_detector.get_cloud_masks(arr)
        else:
            raise ValueError("Constellation has no cloud mask implemented")

        return cloud_mask

    def predict(
        self,
        arr: Any,
        constellation="sentinel-2",
        n_jobs=-1,
        verbose=0,
    ):
        """Return masks for a TxBxHxW array
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
                    delayed(self.predict_single_revisit)(arr, revisit, constellation)
                    for revisit in range(arr.shape[0])
                ],
            )

        return masks
