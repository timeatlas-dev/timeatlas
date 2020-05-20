from typing import List

from timeatlas import TimeSeries

from ._analysis import Analysis
from ._io import IO
from ._processing import Processing


class TimeSeriesDataset(Analysis, IO, Processing):
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

