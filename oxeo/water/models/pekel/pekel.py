from typing import Any

import zarr
from attr import define
from joblib import Parallel, delayed
from satextractor.utils import tqdm_joblib
from tqdm import tqdm

from oxeo.satools.processing import get_mtl_dict, to_toa
from oxeo.water.models import Predictor
from oxeo.water.models.pekel import masks, utils


@define
class PekelPredictor(Predictor):
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
        fs = arr.store.fs

        if "landsat" in constellation:
            mtl_path = "/".join(arr.store.root.split("/")[:-1] + ["metadata"])
            mtl_dict = get_mtl_dict(fs, mtl_path, revisit)
            arr = to_toa(arr[revisit], mtl_dict, constellation)
        else:
            arr = arr[revisit]

        p_bands = utils.pekel_bands(
            arr,
            constellation,
        )
        c_masks = masks.combine_masks(p_bands, cloud_mask=False)
        return c_masks

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
