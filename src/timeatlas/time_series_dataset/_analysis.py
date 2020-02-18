from typing import Any, TYPE_CHECKING
from pandas import DataFrame

if TYPE_CHECKING:
    from .time_series_dataset import TimeSeriesDataset

from timeatlas.abstract.abstract_analysis import AbstractAnalysis


class Analysis(AbstractAnalysis):

    def describe(self) -> DataFrame:
        """
        Describe a TimeSeries with the describe function from Pandas

        Args:
            time_series_dataset: TimeSeriesDataset

        Returns:
            TODO Define return type
        """
        pass

    def compute_resolution(self) \
            -> TimeSeriesDataset:
        """
        Create the series of delta T between each timestamp of a TimeSeries

        Args:
            time_series_dataset: TimeSeriesDataset

        Returns:
            TODO Should it returns a Series of TimeDelta?
        """
        pass

    def compute_duration(self) -> Any:
        """
        Compute the duration of a TimeSeries by giving you its start and ending
        timestamps in a Tuple

        Args:
            time_series_dataset: TimeSeriesDataset

        Returns:
            TODO Add stricter return type once found
        """
        pass

    def compute_correlations(self) -> Any:
        """
        Compute the correlations between all time series present in a
        TimeSeriesDataset

        Args:
            time_series_dataset: TimeSeriesDataset

        Returns:
            TODO Add stricter return type once found
        """
        pass
