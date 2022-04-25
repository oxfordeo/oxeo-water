import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from oxeo.core.models.tile import load_tile_as_dict
from oxeo.core.utils import identity
from oxeo.satools.processing import get_mtl_dict, to_toa
from oxeo.water.models.base import Predictor
from oxeo.water.models.pekel import masks, utils
from oxeo.water.utils import tqdm_joblib


class PekelPredictor(Predictor):
    def __init__(self, fs=None, n_jobs=-1, verbose=1, **kwargs):
        self.fs = fs
        self.n_jobs = n_jobs
        self.verbose = verbose

    def predict_single(self, arr, constellation):
        p_bands = utils.pekel_bands(
            arr,
            constellation,
        )
        c_masks = masks.combine_masks(p_bands, cloud_mask=False)
        return c_masks

    def predict(self, tile_path, revisit, fs):
        """Return masks for a TxBxHxW array
        Args:
            arr (np.ndarray): the array to make predictions in

        Returns:
            [type]: TxHxW masks
        """
        if fs is not None:
            fs_mapper = fs.get_mapper
        else:
            fs_mapper = identity
        arr = load_tile_as_dict(
            fs_mapper=fs_mapper,
            tile_path=tile_path,
            masks=(),
            revisit=revisit,
        )
        arr = arr["image"]
        if "landsat" in tile_path.constellation:
            toa_arrs = []
            for i in range(arr.shape[0]):
                mtl_dict = get_mtl_dict(self.fs, tile_path.metadata_path, i)
                toa_arrs.append(to_toa(arr[i], mtl_dict, tile_path.constellation))
            arr = np.array(toa_arrs)

        with tqdm_joblib(
            tqdm(
                desc="parallel predicting masks on revistis.",
                total=arr.shape[0],
            ),
        ):
            masks = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                [
                    delayed(self.predict_single)(arr[i], tile_path.constellation)
                    for i in range(arr.shape[0])
                ],
            )

        return masks
