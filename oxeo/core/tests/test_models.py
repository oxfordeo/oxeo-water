import pytest

from oxeo.core.models import tile

a_tile = tile.Tile(
    zone=44,
    row="Q",
    min_x=780000,
    min_y=2370000,
    max_x=790000,
    max_y=2380000,
    epsg=32644,
)


@pytest.mark.parametrize(
    "tile_id, expected",
    [
        ("44_Q_10000_78_237", a_tile),
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
