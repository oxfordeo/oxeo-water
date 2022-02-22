from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Callable, Dict, List

import attrs
import numpy as np
import pandas as pd
from attrs import define

from oxeo.water.models import Predictor
from oxeo.water.models.utils import WaterBody, merge_masks_all_constellations


@define
class TestWaterBody(ABC):
    """
    This is an abstract class to use as base for testing a waterbody.
    A TestWaterBody should be instantiated with:
        - waterbody
        - start_date
        - end_date
        - constellation
        - metrics

    Concrete subclasses should implement methods generate_y_true and generate_y_pred
    and optionally can override calculate_metrics.
    """

    waterbody: WaterBody
    start_date: str
    end_date: str
    constellation: str
    metrics: Dict[str, Callable]
    y_true: np.ndarray = attrs.field(init=False)
    y_pred: np.ndarray = attrs.field(init=False)

    def calculate_metrics(self) -> pd.DataFrame:
        """Method to calculate the metrics using y_true and y_pred.
        By default this method assumes y_true and y_pred to be a list of TimeseriesMask.
        It iterates this list and for each element applies all the provided metrics.

        Raises:
            ValueError: _description_

        Returns:
            pd.DataFrame: a dataframe containing a row per timestamp with one column per metric.
                        | timestamp | constellation | metric_1 | metric_2 | etc
        """
        if (self.metrics is None) or not isinstance(self.metrics, dict):
            raise ValueError("Metrics must be a dictionary of callable.")

        if not hasattr(self, "y_true"):
            self.generate_y_true()
        if not hasattr(self, "y_pred"):
            self.generate_y_pred()

        results = defaultdict(list)

        for i, tsm in enumerate(self.y_true):
            for m_name, m_fun in self.metrics.items():
                tsm_pred = self.y_pred[i]
                timestamps_y_true = tsm.data.revisits.values
                timestamps_y_pred = tsm_pred.data.revisits.values
                timestamps_intersect = np.intersect1d(
                    timestamps_y_true, timestamps_y_pred
                )

                timestamps_intersect = timestamps_intersect[
                    (timestamps_intersect >= np.datetime64(self.start_date))
                    & (timestamps_intersect <= np.datetime64(self.end_date))
                ]
                y_true_values = tsm.data.sel(revisits=timestamps_intersect).values
                y_pred_values = tsm_pred.data.sel(revisits=timestamps_intersect).values
                results[m_name].extend(m_fun(y_true_values, y_pred_values))

            results["constellation"].extend(
                [tsm.constellation] * len(timestamps_intersect)
            )
            results["timestamp"].extend(timestamps_intersect)

        df = pd.DataFrame.from_dict(results)
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


@define
class PixelTestWaterBody(TestWaterBody):
    def generate_y_true(self):
        self.y_true = merge_masks_all_constellations(self.waterbody, "cnn")

    def generate_y_pred(self):
        self.y_pred = merge_masks_all_constellations(self.waterbody, "cnn")


@define
class TrainingPixelTestWaterBody(TestWaterBody):
    predictor: Predictor

    def generate_y_true(self):
        self.y_true = merge_masks_all_constellations(self.waterbody, "cnn")

    def generate_y_pred(self):
        self.y_pred = merge_masks_all_constellations(self.waterbody, "cnn")


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
