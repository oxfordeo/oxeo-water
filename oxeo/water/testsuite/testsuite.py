from abc import ABC, abstractmethod
from typing import Callable, Dict, List

import pandas as pd
from attr import define

from oxeo.water.models import Predictor
from oxeo.water.models.utils import TilePath


@define
class TestSuite(ABC):
    """
    This is an abstract class to use as base for different kinds of testsuites.
    A TestSuite should be instantiated with:
        - predictor
        - tile_paths
        - start_date
        - end_date
        - constellations

    an it should return a table with metrics
    """

    predictor: Predictor
    tile_paths: List[TilePath]
    start_date: str
    end_date: str
    constellations: List[str]

    @abstractmethod
    def calculate_metrics(self) -> pd.DataFrame:
        pass


@define
class ScalarTestSuite(TestSuite):
    metrics: Dict[str, Callable]

    def calculate_metrics(self) -> pd.DataFrame:
        pass


@define
class PixelTestSuite(TestSuite):
    metrics: Dict[str, Callable]

    def calculate_metrics(self) -> pd.DataFrame:
        pass
