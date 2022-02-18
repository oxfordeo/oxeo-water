from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Callable, Dict, List

import attrs
import numpy as np
import pandas as pd
from attrs import define

from oxeo.water.models import Predictor
from oxeo.water.models.utils import WaterBody


@define
class TestWaterBody(ABC):
    """
    This is an abstract class to use as base for testing a waterbody.
    A TestWaterBody should be instantiated with:
        - predictor
        - waterbody
        - start_date
        - end_date
        - constellation
        - metrics
    """

    predictor: Predictor
    waterbody: WaterBody
    start_date: str
    end_date: str
    constellation: str
    metrics: Dict[str, Callable]
    y_true: np.ndarray = attrs.field(init=False)
    y_pred: np.ndarray = attrs.field(init=False)
    timestamps: np.ndarray = attrs.field(init=False)

    def calculate_metrics(self) -> pd.DataFrame:
        if (self.metrics is None) or not isinstance(self.metrics, dict):
            raise ValueError("Metrics must be a dictionary of callable.")

        if not hasattr(self, "y_true"):
            self.generate_y_true()
        if not hasattr(self, "y_pred"):
            self.generate_y_pred()
        if not hasattr(self, "timestamps"):
            self.generate_timestamps()

        results = defaultdict(list)
        for m_name, m_fun in self.metrics.items():
            results[m_name] = m_fun(self.y_true, self.y_pred)

        df = pd.DataFrame.from_dict(results)
        df["timestamps"] = self.timestamps
        return df

    @abstractmethod
    def generate_y_true(self):
        """This method generates the y_true from the waterbody,
        dates and constellation.
        """

    @abstractmethod
    def generate_y_pred(self):
        """This method generates the y_pred from the waterbody,
        dates and constellation using the given predictor.
        """

    @abstractmethod
    def generate_timestamps(self):
        """This method generates the timestamps from the waterbody,
        dates and constellation.
        """


@define
class PixelTestWaterBody(TestWaterBody):
    def generate_y_true(self):
        return np.array([1, 0, 0, 1])

    def generate_y_pred(self):
        return np.array([1, 1, 0, 1])

    def generate_timestamps(self):
        return np.arange(np.datetime64("2017-01-01"), np.datetime64("2017-01-05"))


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
    """

    predictor: Predictor
    waterbodies: List[WaterBody]
    start_date: str
    end_date: str
    constellations: List[str]
    metrics: Dict[str, Callable]
    y_true: np.ndarray = attrs.field(init=False)
    y_pred: np.ndarray = attrs.field(init=False)

    def calculate_metrics(self) -> pd.DataFrame:
        if not hasattr(self, "y_true") or not hasattr(self, "y_pred"):
            raise ValueError(
                """y_true and y_pred should be generated before running this method.
                Run generate_y_true() and generate_y_pred() to do so.
                """
            )
        if (self.metrics is None) or not isinstance(self.metrics, dict):
            raise ValueError("Metrics must be a dictionary of callable.")

        results = defaultdict(list)
        for m_name, m_fun in self.metrics.items():
            results[m_name] = m_fun(self.y_true, self.y_pred)

        area_ids = [w.area_id for w in self.waterbodies]
        df = pd.DataFrame.from_dict(results)
        df["area_id"] = area_ids
        return df

    @abstractmethod
    def generate_y_true(self):
        """This method generates the y_true from the waterbodies,
        dates and constellation.
        """

    @abstractmethod
    def generate_y_pred(self):
        """This method generates the y_pred from the waterbodies,
        dates and constellation using the given predictor.
        """


@define
class ScalarTestSuite(TestSuite):
    metrics: Dict[str, Callable]

    def generate_y_true(self):
        """This method generates the y_true from the tile_paths, dates and constellation.
        the y_true property will be used to calculate the metrics
        """
        self.y_true = None

    def generate_y_pred(self):
        """This method generates the y_pred from the tile_paths, dates and constellation
        using the given predictor. The y_pred property will be used to calculate the metrics
        """
        self.y_pred = None

    def calculate_metrics(self) -> pd.DataFrame:
        if not hasattr(self, "y_true") or not hasattr(self, "y_pred"):
            raise ValueError(
                "y_true and y_pred should be generated before running this method."
            )


@define
class PixelTestSuite(TestSuite):
    metrics: Dict[str, Callable]

    def calculate_metrics(self) -> pd.DataFrame:
        if (self.y_true is None) or (self.y_pred is None):
            raise ValueError("Generate y_true and y_pred before running this method.")
