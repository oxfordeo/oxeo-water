import numpy as np

from oxeo.water.metrics import segmentation_area


import pytest


segmentation = np.array([[0, 0, 0, 1], [0, 1, 1, 1], [0, 0, 1, 1]])
segmentation_seq = np.array(
    [
        [[0, 0, 0, 1], [0, 1, 1, 1], [0, 0, 1, 1]],
        [[0, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 1]],
    ]
)


def test_segmentation_area_with_pixel_unit():
    assert segmentation_area(segmentation, "pixel") == 6


@pytest.mark.parametrize(
    "seg,unit,resolution, expected",
    [
        (segmentation, "meter", 10, 60),
        (segmentation, "meter", 0.5, 3),
        (segmentation, "meter", 20, 120),
    ],
)
def test_segmentation_area_with_meters_unit(seg, unit, resolution, expected):
    assert segmentation_area(seg, unit, resolution) == expected


@pytest.mark.parametrize(
    "seg,unit,resolution, expected",
    [(segmentation_seq, "pixel", None, 10), (segmentation_seq, "meter", 10, 100),],
)
def test_segmentation_area_with_sequence(seg, unit, resolution, expected):
    assert segmentation_area(seg, unit, resolution) == expected
