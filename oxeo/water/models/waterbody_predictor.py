from datetime import datetime
from typing import Any, List

import attrs
import numpy as np
import zarr
from attrs import define

from oxeo.core.models.timeseries import TimeseriesMask, merge_masks_all_constellations
from oxeo.core.models.waterbody import WaterBody
from oxeo.water.models.base import Predictor
from oxeo.water.models.factory import model_factory
from oxeo.water.utils.utils import identity


@define
class WaterBodyPredictor:
    fs: Any
    model_name: str
    revisit_chunk_size: int
    ckpt_path: str = attrs.field(default=None)
    batch_size: int = attrs.field(default=None)
    bands: list[str] = attrs.field(default=None)
    target_size: int = attrs.field(default=None)
    predictor: Predictor = attrs.field(init=False)

    def __attrs_post_init__(self):
        self.predictor = model_factory(self.model_name).predictor(
            ckpt_path=self.ckpt_path,
            fs=self.fs,
            batch_size=self.batch_size,
            bands=self.bands,
            target_size=self.target_size,
        )

    def predict(
        self, waterbody: WaterBody, start_date: str, end_date: str, fs: Any = None
    ) -> List[TimeseriesMask]:
        tile_paths = waterbody.paths
        sdt = np.datetime64(datetime.strptime(start_date, "%Y-%m-%d"))
        edt = np.datetime64(datetime.strptime(end_date, "%Y-%m-%d"))

        if fs is not None:
            fs_mapper = fs.get_mapper
        else:
            fs_mapper = identity
        for t_path in tile_paths:
            mask_list = []
            timestamps = zarr.open(fs_mapper(t_path.timestamps_path), "r")[:]
            timestamps = np.array(
                [np.datetime64(datetime.fromisoformat(el)) for el in timestamps],
            )
            date_overlap = np.where((timestamps >= sdt) & (timestamps <= edt))[0]
            min_idx = date_overlap.min()
            max_idx = date_overlap.max() + 1

            mask_path = f"{t_path.mask_path}/{self.model_name}"

            for i in range(min_idx, max_idx, self.revisit_chunk_size):
                revisit_masks = self.predictor.predict(
                    t_path,
                    revisit=slice(i, min(i + self.revisit_chunk_size, max_idx)),
                    fs=fs,
                )
                mask_list.append(revisit_masks)
            masks = np.vstack(mask_list)

            time_shape = timestamps.shape[0]
            geo_shape = masks.shape[1:]
            output_shape = (time_shape, *geo_shape)

            mask_arr = zarr.open_array(
                fs_mapper(mask_path),
                "a",
                shape=output_shape,
                chunks=(1, 1000, 1000),
                dtype=np.uint8,
            )
            mask_arr.resize(*output_shape)
            mask_arr.attrs["max_filled"] = int(max_idx)

            # write data to archive
            mask_arr[min_idx:max_idx, ...] = masks

        return merge_masks_all_constellations(waterbody, self.model_name)
