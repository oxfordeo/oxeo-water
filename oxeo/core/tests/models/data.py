import numpy as np
import zarr

from oxeo.core.models import tile

zeros_zarr_arr = zarr.zeros((2, 4, 10, 10), chunks=(1, 1, 10, 10), dtype="i4")
small_zeros_zarr_arr = zarr.zeros((2, 4, 5, 5), chunks=(1, 1, 5, 5), dtype="i4")
zeros_zarr_mask = zarr.zeros((2, 10, 10), chunks=(1, 10, 10), dtype="i4")

sample = {
    "image": zeros_zarr_arr[:],
    "mask": zeros_zarr_mask[:][np.newaxis, ...],
}

two_bands_sample = {
    "image": zeros_zarr_arr[:, :2],
    "mask": zeros_zarr_mask[:][np.newaxis, ...],
}

one_revisit_sample = {
    "image": zeros_zarr_arr[:1],
    "mask": zeros_zarr_mask[:1][np.newaxis, ...],
}

sample_without_mask = {
    "image": zeros_zarr_arr[:],
}

square_tile = tile.Tile(
    zone=44,
    row="Q",
    min_x=780000,
    min_y=2370000,
    max_x=790000,
    max_y=2380000,
    epsg=32644,
)

non_square_tile = tile.Tile(
    zone=44,
    row="Q",
    min_x=780000,
    min_y=2360000,  # changed
    max_x=790000,
    max_y=2380000,
    epsg=32644,
)

sentinel2_tile_path = tile.TilePath(square_tile, "sentinel-2", "gs://oxeo-water/test")
landsat8_tile_path = tile.TilePath(square_tile, "landsat-8", "gs://oxeo-water/test")
