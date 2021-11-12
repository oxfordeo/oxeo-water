import numpy as np
import pytest

from oxeo.water.metrics import pearson, segmentation_area

segmentation = np.array([[0, 0, 0, 1], [0, 1, 1, 1], [0, 0, 1, 1]])
segmentation_seq = np.array(
    [
        [[0, 0, 0, 1], [0, 1, 1, 1], [0, 0, 1, 1]],
        [[0, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 1]],
    ]
)

x = [6, 10, 2, 1, 56]
y = [62, 108, 23, 11, 568]


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
    [
        (segmentation_seq, "pixel", None, [6, 4]),
        (segmentation_seq, "meter", 10, [60, 40]),
    ],
)
def test_segmentation_area_with_sequence(seg, unit, resolution, expected):
    assert (segmentation_area(seg, unit, resolution) == expected).all()


def test_pearson():
    assert pearson(x, y) == pytest.approx(0.99, 0.1)
