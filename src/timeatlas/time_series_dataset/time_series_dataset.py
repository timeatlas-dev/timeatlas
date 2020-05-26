from typing import List, Any, NoReturn

from pandas import DataFrame

from timeatlas import TimeSeries
from timeatlas.utils import ensure_dir, to_pickle

from timeatlas.abstract import (
    AbstractAnalysis,
    AbstractOutputText,
    AbstractOutputPickle,
    AbstractProcessing
)


class TimeSeriesDataset(AbstractAnalysis, AbstractProcessing, AbstractOutputText, AbstractOutputPickle):
    """ Defines a set of time series

    A TimeSeriesDataset represent a set of TimeSeries
    objects.

    """

    def __init__(self, data: List[TimeSeries] = None):
        if data is None:
            self.data = []
        else:
            self.data = data

    # Methods
    # =======

    def add(self, time_series: TimeSeries):
        self.data.append(time_series)

    def remove(self, index):
        del self.data[index]

    def len(self):
        return len(self.data)

    # =============================================
    # Analysis
    # =============================================

    def describe(self) -> DataFrame:
        """
        Describe a TimeSeries with the describe function from Pandas

        Returns:
            TODO Define return type
        """
        pass

    def compute_resolution(self):
        """
        Create the series of delta T between each timestamp of a TimeSeries

        Returns:
            TODO Should it returns a Series of TimeDelta?
        """
        pass

    def compute_duration(self) -> Any:
        """
        Compute the duration of a TimeSeries by giving you its start and ending
        timestamps in a Tuple

        Returns:
            TODO Add stricter return type once found
        """
        pass

    def compute_correlations(self) -> Any:
        """
        Compute the correlations between all time series present in a
        TimeSeriesDataset

        Returns:
            TODO Add stricter return type once found
        """
        pass

    # =============================================
    # Processing
    # =============================================

    def resample(self, by: str) -> Any:
        """
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
        - upsampling
        - downsampling
        """
        pass

    def interpolate(self, method: str) -> Any:
        """
        "Intelligent" interpolation in function of the data unit etc.
        """
        pass

    def normalize(self, method: str) -> Any:
        """
        Normalize a dataset
        """
        pass

    def synchronize(self):
        """
        Synchronize the timestamps so that they are the same for all time
        series in the TimeSeriesDataset
        """
        pass

    def unify(self):
        """
        Put all time series in a matrix iff they all have the same length
        """
        pass

    # =============================================
    # IO
    # =============================================

    def to_text(self, path: str) -> NoReturn:
        ensure_dir(path)
        for i, ts in enumerate(self.data):
            ts_path = "{}/{}".format(path, i)
            ensure_dir(ts_path)
            ts.to_text(ts_path)

    def to_pickle(self, path: str) -> NoReturn:
        to_pickle(self, path)
