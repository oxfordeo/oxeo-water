from datetime import datetime
from typing import Any, Dict, List, Tuple, Union

import geopandas as gpd
import numpy as np
import torch
import zarr
from attr import define
from pyproj import CRS
from pystac.extensions.eo import Band
from satextractor.models import Tile
from satextractor.models.constellation_info import BAND_INFO
from satextractor.tiler import split_region_in_utm_tiles
from shapely.geometry import MultiPolygon, Polygon
from torchvision.transforms.functional import InterpolationMode, resize
from zarr.errors import ArrayNotFoundError

from oxeo.core.logging import logger
from oxeo.water.models.factory import model_factory
from oxeo.water.utils.utils import identity


@define(frozen=True)
class TilePath:
    tile: Tile
    constellation: str
    root: str = "gs://oxeo-water/prod"

    @property
    def path(self):
        return f"{self.root}/{self.tile.id}/{self.constellation}"

    @property
    def timestamps_path(self):
        return f"{self.path}/timestamps"

    @property
    def data_path(self):
        return f"{self.path}/data"

    @property
    def mask_path(self):
        return f"{self.path}/mask"

    @property
    def metadata_path(self):
        return f"{self.path}/metadata"


def tile_from_id(id):
    zone, row, bbox_size_x, xloc, yloc = id.split("_")
    bbox_size_x, xloc, yloc = int(bbox_size_x), int(xloc), int(yloc)
    bbox_size_y = bbox_size_x
    min_x = xloc * bbox_size_x
    min_y = yloc * bbox_size_y
    max_x = min_x + bbox_size_x
    max_y = min_y + bbox_size_y
    south = row < "N"
    epsg = CRS.from_dict({"proj": "utm", "zone": zone, "south": south}).to_epsg()
    return Tile(
        zone=int(zone),
        row=row,
        min_x=min_x,
        min_y=min_y,
        max_x=max_x,
        max_y=max_y,
        epsg=epsg,
    )


def get_band_list(constellation: str) -> List[str]:
    BAND_INFO["sentinel-1"] = {
        "B1": {"band": Band.create(name="B1", common_name="VV")},
        "B2": {"band": Band.create(name="B2", common_name="VH")},
    }
    return [b["band"].common_name for b in BAND_INFO[constellation].values()]


def get_patch_size(patch_paths: List[TilePath]) -> int:  # in pixels
    # TODO: Probably unnecessary to load all patches for this,
    # could just assume they're the same size
    sizes = []
    for patch in patch_paths:
        arr_path = f"{patch.path}/data"
        logger.info(f"Loading to check size {arr_path=}")
        z = zarr.open(arr_path, "r")
        x, y = z.shape[2:]
        assert x == y, "Must use square patches"
        sizes.append(x)
    assert len(set(sizes)) == 1, "All sizes must be the same"
    return sizes[0]


def get_tile_size(tiles: List[Tile]) -> int:  # in metres
    sizes = []
    for tile in tiles:
        x, y = tile.bbox_size_x, tile.bbox_size_y
        assert x == y, "Must use square tiles"
        sizes.append(x)
    assert len(set(sizes)) == 1, "All sizes must be the same"
    return sizes[0]


def make_paths(tiles, constellations, root_dir) -> List[TilePath]:
    return [
        TilePath(tile=tile, constellation=cons, root=root_dir)
        for tile in tiles
        for cons in constellations
    ]


def get_tiles(
    geom: Union[Polygon, MultiPolygon, gpd.GeoSeries, gpd.GeoDataFrame]
) -> list[Tile]:
    try:
        geom = geom.unary_union
    except AttributeError:
        pass
    return split_region_in_utm_tiles(region=geom, bbox_size=10000)


def get_all_paths(
    gdf: gpd.GeoDataFrame,
    constellations: list[str],
    root_dir: str = "gs://oxeo-water/prod",
) -> list[TilePath]:
    all_tiles = get_tiles(gdf)
    all_tilepaths = make_paths(all_tiles, constellations, root_dir)
    logger.info(
        f"All tiles for the supplied geometry: {[t.path for t in all_tilepaths]}"
    )
    return all_tilepaths


def load_tile(
    fs_mapper,
    tile_path: TilePath,
    masks: Tuple[str, ...] = (),
    revisit: slice = None,
    bands: Tuple[str, ...] = None,
) -> torch.Tensor:
    if bands is not None:
        band_common_names = get_band_list(tile_path.constellation)
        band_indices = [band_common_names.index(b) for b in bands]
    else:
        band_common_names = get_band_list(tile_path.constellation)
        band_indices = list(range(0, len(band_common_names)))

    sample = {}
    arr = zarr.open_array(fs_mapper(tile_path.data_path), mode="r")
    logger.info(f"{arr.shape=}; {revisit=}, {band_indices=}")
    arr = arr.oindex[revisit, band_indices].astype(np.int16)
    logger.info(f"{arr.shape=}")
    for mask in masks:
        mask_arr = zarr.open_array(
            fs_mapper(f"{tile_path.mask_path}/{mask}"), mode="r"
        )[revisit].astype(np.int8)
        mask_arr = mask_arr[np.newaxis, ...]
        sample[mask] = torch.as_tensor(mask_arr)

    sample["image"] = torch.as_tensor(arr)

    return sample


def resize_sample(
    sample: Union[torch.Tensor, Dict[str, torch.Tensor]],
    target_size: int = None,
    interpolation=InterpolationMode.NEAREST,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """Resize sample to target

    Args:
        sample (Union[torch.Tensor, Dict[str, torch.Tensor]]): Can be a tensor or dict of tensors
        target_size (int, optional): Target size. Defaults to None.
        interpolation ([type], optional): Only used when sample is torch.Tensor.
                                          Defaults to InterpolationMode.NEAREST.

    Returns:
        torch.Tensor: the resampled tensor or dict of tensors
    """
    logger.debug(f"Resizing sample to {target_size}")
    if target_size is not None:
        if isinstance(sample, dict):
            resized_sample = {}
            for key in sample.keys():
                if key == "image":
                    resized_sample[key] = resize(
                        sample[key], target_size, InterpolationMode.BILINEAR
                    )
                else:
                    resized_sample[key] = resize(
                        sample[key], target_size, InterpolationMode.NEAREST
                    )
        elif isinstance(sample, torch.Tensor):
            resized_sample = resize(sample, target_size, interpolation)
    return resized_sample


def load_tile_and_resize(
    fs_mapper,
    tile_path: TilePath,
    masks,
    revisit: int = None,
    bands=None,
    target_size=None,
):
    sample = load_tile(
        fs_mapper,
        tile_path,
        masks=masks,
        revisit=revisit,
        bands=bands,
    )
    sample = resize_sample(sample, target_size)
    return sample


def predict_tile(
    tile_path: TilePath,
    model_name: str,
    ckpt_path: str,
    target_size: int,
    bands: list[str],
    cnn_batch_size: int,
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

    logger.info("Loading model")
    predictor = model_factory(model_name).predictor(
        ckpt_path=ckpt_path,
        fs=fs,
        batch_size=cnn_batch_size,
        bands=bands,
        target_size=target_size,
    )

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
