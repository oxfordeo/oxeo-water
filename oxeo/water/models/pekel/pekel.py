from typing import Any

from attr import define
from joblib import Parallel, delayed
from satextractor.utils import tqdm_joblib
from tqdm import tqdm

from oxeo.water.models import Predictor
from oxeo.water.models.pekel import masks, utils


@define
class PekelPredictor(Predictor):
    def predict_single_revisit(
        self,
        arr: Any,
        revisit: int,
        constellation="sentinel-2",
        compute=False,
    ):
        """Return mask for a BxHxW array
        Args:
            arr (np.ndarray): the array to make predictions in

        Returns:
            [type]: HxW mask
        """
        arr = arr[revisit]
        if compute:
            arr = arr.compute()

        p_bands = utils.pekel_bands(arr, constellation)
        c_masks = masks.combine_masks(p_bands, False)
        return c_masks

    def predict(
        self,
        arr: Any,
        constellation="sentinel-2",
        compute=False,
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
                    delayed(self.predict_single_revisit)(
                        arr, revisit, constellation, compute
                    )
                    for revisit in range(arr.shape[0])
                ],
            )

        return masks
