import numpy as np
from tqdm import tqdm

from oxeo.core.models.tile import load_tile
from oxeo.core.utils import identity
from oxeo.satools.processing import get_mtl_dict, to_toa
from oxeo.water.models.base import Predictor
from oxeo.water.models.pekel import masks, utils


class PekelPredictor(Predictor):
    def __init__(self, fs=None, **kwargs):
        self.fs = fs

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
        arr = load_tile(
            fs_mapper=fs_mapper,
            tile_path=tile_path,
            masks=(),
            revisit=revisit,
        )
        arr = arr["image"].numpy()
        if "landsat" in tile_path.constellation:
            toa_arrs = []
            for i in range(arr.shape[0]):
                mtl_dict = get_mtl_dict(self.fs, tile_path.metadata_path, i)
                toa_arrs.append(to_toa(arr[i], mtl_dict, tile_path.constellation))
            arr = np.array(toa_arrs)

        preds = []
        for patch in tqdm(range(0, arr.shape[0])):
            p_bands = utils.pekel_bands(
                arr[patch],
                tile_path.constellation,
            )
            c_masks = masks.combine_masks(p_bands, cloud_mask=False)
            preds.append(c_masks)
        return np.array(preds)
