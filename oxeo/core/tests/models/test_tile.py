import numpy as np
import pytest
import zarr

from oxeo.core.models import tile
from oxeo.core.utils import identity


def compare_exact(first, second):
    """Assert whether two dicts of arrays are exactly equal"""
    assert first.keys() == second.keys()
    [np.testing.assert_array_equal(first[key], second[key]) for key in first]


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


# tile_id tests


@pytest.mark.parametrize(
    "tile_id, expected",
    [
        ("44_Q_10000_78_237", square_tile),
    ],
)
def test_tile_from_id_ok(tile_id, expected):
    assert tile.tile_from_id(tile_id) == expected


@pytest.mark.parametrize(
    "tile_id, expected",
    [
        ("44_Q_10000_78", ValueError),
        (441000078, AttributeError),
    ],
)
def test_tile_from_id_error(tile_id, expected):
    with pytest.raises(expected):
        tile.tile_from_id(tile_id)


# get_patch_size tests


@pytest.mark.parametrize(
    "tile_paths, expected",
    [
        ([sentinel2_tile_path, sentinel2_tile_path], 10),
    ],
    ids=["same_patch_size"],
)
def test_get_patch_size_ok(mocker, tile_paths, expected):
    mocker.patch("zarr.open", return_value=zeros_zarr_arr)
    assert tile.get_patch_size(tile_paths) == expected


@pytest.mark.parametrize(
    "tile_paths, expected",
    [
        ([sentinel2_tile_path, landsat8_tile_path], AssertionError),
        ([], AssertionError),
    ],
    ids=["different_patch_size", "empty_paths"],
)
def test_get_patch_size_error(mocker, tile_paths, expected):
    mocker.patch("zarr.open", side_effect=[zeros_zarr_arr, small_zeros_zarr_arr])
    with pytest.raises(expected):
        tile.get_patch_size(tile_paths)


# get_tile_size tests


@pytest.mark.parametrize(
    "tiles, expected",
    [
        ([square_tile], 10000),
    ],
)
def test_get_tile_size_ok(tiles, expected):
    assert tile.get_tile_size(tiles) == expected


@pytest.mark.parametrize(
    "tiles, expected",
    [
        ([non_square_tile], AssertionError),
        ([square_tile, non_square_tile], AssertionError),
    ],
)
def test_get_tile_size_error(tiles, expected):
    with pytest.raises(expected):
        tile.get_tile_size(tiles)


# make_paths tests


@pytest.mark.parametrize(
    "tiles, constellations, root_dir, expected",
    [
        ([square_tile], ["sentinel-2"], "gs://oxeo-water/test", [sentinel2_tile_path]),
        (
            [square_tile],
            ["sentinel-2", "landsat-8"],
            "gs://oxeo-water/test",
            [sentinel2_tile_path, landsat8_tile_path],
        ),
        (
            [square_tile, square_tile],
            ["sentinel-2", "landsat-8"],
            "gs://oxeo-water/test",
            [
                sentinel2_tile_path,
                landsat8_tile_path,
                sentinel2_tile_path,
                landsat8_tile_path,
            ],
        ),
    ],
)
def test_make_paths_ok(tiles, constellations, root_dir, expected):
    assert tile.make_paths(tiles, constellations, root_dir) == expected


# load_tile_as_dict tests


@pytest.mark.parametrize(
    "fs_mapper, tile_path, masks, revisit, bands, expected",
    [
        (
            identity,
            sentinel2_tile_path,
            ("mask",),
            slice(None),
            ("blue", "green"),
            two_bands_sample,
        ),
        (
            identity,
            sentinel2_tile_path,
            ("mask",),
            slice(1),
            ("coastal", "blue", "green", "red"),
            one_revisit_sample,
        ),
        (
            identity,
            sentinel2_tile_path,
            (),
            slice(None),
            ("coastal", "blue", "green", "red"),
            sample_without_mask,
        ),
    ],
    ids=["reduced_bands", "reduced_revisits", "no_mask"],
)
def test_load_tile_as_dict_ok(
    mocker, fs_mapper, tile_path, masks, revisit, bands, expected
):
    mocker.patch("zarr.open_array", side_effect=[zeros_zarr_arr, zeros_zarr_mask])
    sample = tile.load_tile_as_dict(fs_mapper, tile_path, masks, revisit, bands)
    compare_exact(sample, expected)


@pytest.mark.parametrize(
    "fs_mapper, tile_path, masks, revisit, bands, expected",
    [
        (
            identity,
            sentinel2_tile_path,
            (),
            slice(0, 2),
            ("NON_EXISTING_BAND", "red"),
            ValueError,
        )
    ],
    ids=["non_existing_band"],
)
def test_load_tile_as_dict_error(
    mocker, fs_mapper, tile_path, masks, revisit, bands, expected
):
    mocker.patch("zarr.open_array", return_value=zeros_zarr_arr)
    with pytest.raises(expected):
        tile.load_tile_as_dict(fs_mapper, tile_path, masks, revisit, bands)
