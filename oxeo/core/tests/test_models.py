import pytest

from oxeo.core.models import tile

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
    min_y=2360000,
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
