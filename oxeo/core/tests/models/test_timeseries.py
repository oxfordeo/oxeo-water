import os
import pickle

import numpy as np
import pytest
import zarr
from sqlalchemy import create_engine

from oxeo.core.models import timeseries
from oxeo.core.models.data import data2gdf, fetch_water_list
from oxeo.core.models.waterbody import get_waterbodies


@pytest.fixture
def victoria_tsm():
    path_to_current_file = os.path.realpath(__file__)
    current_directory = os.path.dirname(path_to_current_file)
    with open(f"{current_directory}/victoria_tsm.pickle", "rb") as f:
        victoria_tsm = pickle.load(f)

    return victoria_tsm


@pytest.fixture
def setup_database():
    """Fixture to set up connection to database"""
    DB_USER = os.environ.get("DB_USER")
    DB_PASSWORD = os.environ.get("DB_PASSWORD")
    DB_HOST = os.environ.get("DB_HOST")
    engine = create_engine(
        f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:5432/geom"
    )
    yield engine


# fixture that gets a real waterbody from remote
@pytest.fixture(params=[[25906117]])
def waterbody(request, setup_database):
    engine = setup_database
    data = fetch_water_list(request.param, engine)
    gdf = data2gdf(data)
    constellations = ["sentinel-2", "landsat-8"]
    root_dir = "/home/fran/tiles"
    waterbodies = get_waterbodies(gdf, constellations, root_dir)
    return waterbodies[0]


# test merge_masks_one_constellation


@pytest.mark.slow
def test_merge_mask_one_constellation_victoria(waterbody, victoria_tsm):
    tsm = timeseries.merge_masks_one_constellation(waterbody, "landsat-8", "cnn")
    np.testing.assert_array_equal(tsm.data, victoria_tsm.data)
    assert tsm.constellation == victoria_tsm.constellation
    assert tsm.resolution == victoria_tsm.resolution


@pytest.mark.parametrize(
    "constellation, mask, expected",
    [
        ("wrong_constellation", "cnn", ValueError),
        ("landsat-8", "what_is_this_mask", zarr.errors.ArrayNotFoundError),
    ],
    ids=["nonexistent_constellation", "nonexistent_mask"],
)
@pytest.mark.slow
def test_merge_mask_one_constellation_error(waterbody, constellation, mask, expected):
    with pytest.raises(expected):
        timeseries.merge_masks_one_constellation(waterbody, constellation, mask)


# test merge_masks_all_constellations


@pytest.mark.slow
def test_merge_masks_all_constellations_ok(waterbody, victoria_tsm):
    tsms = timeseries.merge_masks_all_constellations(waterbody, "cnn")
    constellations = {tsm.constellation for tsm in tsms}
    resolutions = {tsm.resolution for tsm in tsms}
    assert len(tsms) == 2  # sentinel-2 and landast-8
    assert constellations == {"sentinel-2", "landsat-8"}
    assert resolutions == {10, 15}


@pytest.mark.parametrize(
    "mask, expected",
    [
        ("what_is_this_mask", AssertionError),
    ],
    ids=["nonexistent_mask"],
)
@pytest.mark.slow
def test_merge_masks_all_constellations_error(waterbody, mask, expected):
    with pytest.raises(expected):
        timeseries.merge_masks_all_constellations(waterbody, mask)
