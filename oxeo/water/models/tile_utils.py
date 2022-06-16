from datetime import datetime
from typing import Any, Dict, List, Union

import numpy as np
import torch
import zarr
from torchvision.transforms.functional import InterpolationMode, resize
from zarr.errors import ArrayNotFoundError

from oxeo.core.constants import RESOLUTION_INFO
from oxeo.core.logging import logger
from oxeo.core.models.tile import TilePath, load_tile_as_dict
from oxeo.core.utils import identity
from oxeo.satools.io import strdates_to_datetime


def pad_sample(
    sample: Union[torch.Tensor, Dict[str, Union[torch.Tensor, np.ndarray]]],
    pad_to: int = None,
) -> Union[torch.Tensor, Dict[str, Union[torch.Tensor, np.ndarray]]]:
    """Pad sample to target
    Args:
        sample (Union[torch.Tensor, Dict[str, torch.Tensor]]): Can be a tensor or dict of tensors
        pad_to (int): The the final size will be a multiplier of this number.
    Returns:
        torch.Tensor: the resampled tensor or dict of tensors
    """

    if isinstance(sample, dict):
        for key in sample.keys():
            pad_top = abs(sample[key].shape[-2] % -pad_to)
            pad_right = abs(sample[key].shape[-1] % -pad_to)
            sample[key] = torch.nn.functional.pad(
                sample[key], pad=(0, pad_right, 0, pad_top)
            )
    elif isinstance(sample, torch.Tensor):
        pad_top = abs(sample.shape[-2] % -pad_to)
        pad_right = abs(sample.shape[-1] % -pad_to)
        sample = torch.nn.functional.pad(sample, pad=(0, pad_right, 0, pad_top))
    return sample


def get_target_size(
    sample: torch.Tensor, sample_resolution: int, target_resolution
) -> int:
    """Get target size for resampling a sample to a target resolution
    Args:
        sample (torch.Tensor): The sample to be resampled
        sample_resolution (int): The resolution of the sample
        target_resolution (int): The resolution to which the sample should be resampled
    Returns:
        int: The target size
    """
    min_edge = min(sample.shape[-2:])
    if sample_resolution != target_resolution:
        if sample_resolution > target_resolution:
            target_size = min_edge * (sample_resolution / target_resolution / 2)
        else:
            target_size = min_edge * (sample_resolution / target_resolution) * 2
    else:
        target_size = min_edge
    return target_size


def resize_sample(
    sample: Union[torch.Tensor, Dict[str, Union[torch.Tensor, np.ndarray]]],
    sample_resolution: int = RESOLUTION_INFO["sentinel-2"],
    target_resolution: int = RESOLUTION_INFO["sentinel-2"],
    interpolation=InterpolationMode.NEAREST,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """Resize sample to target
    Args:
        sample (Union[torch.Tensor, Dict[str, torch.Tensor]]): Can be a tensor or dict of tensors
        sample_resolution (int): The sample resolution.
        target_resolution (int): The target resolution.
        interpolation ([type], optional): Only used when sample is torch.Tensor.
                                          Defaults to InterpolationMode.NEAREST.
    Returns:
        torch.Tensor: the resampled tensor or dict of tensors
    """

    if isinstance(sample, dict):
        resized_sample = {}
        for key in sample.keys():
            target_size = get_target_size(
                sample[key], sample_resolution, target_resolution
            )
            if key == "image":
                resized_sample[key] = resize(
                    torch.as_tensor(sample[key]),
                    target_size,
                    InterpolationMode.BILINEAR,
                )

            else:
                resized_sample[key] = resize(
                    torch.as_tensor(sample[key]),
                    target_size,
                    InterpolationMode.NEAREST,
                )
    elif isinstance(sample, torch.Tensor):
        target_size = get_target_size(sample, sample_resolution, target_resolution)
        resized_sample = resize(sample, target_size, interpolation)
    return resized_sample


def load_tile_as_dict_and_resize(
    fs_mapper,
    tile_path: TilePath,
    masks,
    revisit: int = None,
    bands=None,
    target_size=None,
):
    sample = load_tile_as_dict(
        fs_mapper,
        tile_path,
        masks=masks,
        revisit=revisit,
        bands=bands,
    )
    sample = resize_sample(sample, target_size)
    return sample


def predict_tile_revisits(
    tile_path: TilePath,
    dates: List[str],
    predictor: Any,
    fs: Any = None,
    gpu: int = 0,
) -> np.ndarray:
    dates = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
    if fs is not None:
        fs_mapper = fs.get_mapper
    else:
        fs_mapper = identity

    timestamps = zarr.open(fs_mapper(tile_path.timestamps_path), "r")[:]
    timestamps = [d for d in strdates_to_datetime(timestamps)]

    date_indices = [timestamps.index(d) for d in dates]

    if gpu > 0:
        import torch

        cuda = torch.cuda.is_available()
        logger.info(f"CUDA available: {cuda}")

    revisit_masks = predictor.predict(
        tile_path,
        revisit=date_indices,
        fs=fs,
    )
    return revisit_masks


def predict_tile(
    tile_path: TilePath,
    model_name: str,
    predictor: Any,
    revisit_chunk_size: int,
    start_date: str,
    end_date: str,
    fs: Any = None,
    overwrite: bool = False,
    gpu: int = 0,
) -> np.ndarray:
    sdt = np.datetime64(datetime.strptime(start_date, "%Y-%m-%d"))
    edt = np.datetime64(datetime.strptime(end_date, "%Y-%m-%d"))
    if fs is not None:
        fs_mapper = fs.get_mapper
    else:
        fs_mapper = identity

    timestamps = zarr.open(fs_mapper(tile_path.timestamps_path), "r")[:]
    timestamps = np.array(
        [np.datetime64(datetime.fromisoformat(el)) for el in timestamps],
    )
    date_overlap = np.where((timestamps >= sdt) & (timestamps <= edt))[0]
    min_idx = date_overlap.min()
    max_idx = date_overlap.max() + 1

    logger.info(f"From overlap with imagery and dates entered: {min_idx=}, {max_idx=}")

    mask_path = f"{tile_path.mask_path}/{model_name}"

    if not overwrite:
        # check existing masks and only do new ones
        try:
            mask_arr = zarr.open_array(fs_mapper(mask_path), "r")
            prev_max_idx = int(mask_arr.attrs["max_filled"])
            min_idx = prev_max_idx + 1
            logger.warning(f"Found {prev_max_idx=}, set {min_idx=}")
        except ArrayNotFoundError:
            logger.warning("Set overwrite=False, but there was no existing array")
        except KeyError:
            logger.warning(
                "Set overwrite=False, but attrs['max_filled'] had not been set"
            )

    if min_idx >= max_idx:
        logger.warning("min_idx is >= max_idx: nothing to do, skipping")
        # TODO This should rather return the last date IN the array
        # Otherwise this will always just be 2100-01-01 or something
        return end_date, end_date

    if gpu > 0:
        import torch

        cuda = torch.cuda.is_available()
        logger.info(f"CUDA available: {cuda}")

    mask_list = []
    for i in range(min_idx, max_idx, revisit_chunk_size):
        logger.info(
            f"creating mask for {tile_path.path}, revisits {i} to "
            f"{min(i + revisit_chunk_size,max_idx)} of {max_idx}"
        )
        revisit_masks = predictor.predict(
            tile_path,
            revisit=slice(i, min(i + revisit_chunk_size, max_idx)),
            fs=fs,
        )
        mask_list.append(revisit_masks)
    masks = np.vstack(mask_list)

    time_shape = timestamps.shape[0]
    geo_shape = masks.shape[1:]
    output_shape = (time_shape, *geo_shape)

    logger.info(f"Saving mask to {mask_path}")
    logger.info(f"Output zarr shape: {output_shape}")

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

    written_start = np.datetime_as_string(timestamps[min_idx], unit="D")
    written_end = np.datetime_as_string(timestamps[max_idx - 1], unit="D")

    return written_start, written_end
