from typing import List

from timeatlas.time_series.time_series import TimeSeries

from ._analysis import Analysis
from ._io import IO
from ._processing import Processing


class TimeSeriesDataset(Analysis, IO, Processing):
    """ Defines a set of time series

    A TimeSeriesDataset represent an unordered, immutable set of TimeSeries
    objects.

    """

    def __init__(self, data: List[TimeSeries]):
        self._data = data
        # TODO Should it make a sanity check?

    # Methods
    # =======

    def add(self, time_series: TimeSeries) -> 'TimeSeriesDataset':
        raise NotImplementedError

    def remove(self, position) -> 'TimeSeriesDataset':
        raise NotImplementedError

