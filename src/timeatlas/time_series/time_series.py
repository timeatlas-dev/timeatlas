from typing import Dict
from pandas import Series, DatetimeIndex

from ._analysis import Analysis
from ._io import IO
from ._processing import Processing
from ._utils import Utils


class TimeSeries(IO, Analysis, Processing):
    """ Defines a time series

    A TimeSeries object is a series of time indexed values.

    Attributes:
        series: An optional Pandas Series
        metadata: An optional Dict storing metadata about this TimeSeries
    """

    def __init__(self, series: Series = None, metadata: Dict = None):

        if series is not None:
            # Check if values have a DatetimeIndex
            assert isinstance(series.index, DatetimeIndex), \
                'Values must be indexed with a DatetimeIndex.'

            # Check if the length is bigger than one
            assert len(series) >= 1, 'Values must have at least one values.'

            # Give a default name to the series (for the CSV output)
            series.name = "values"

        # Create the TimeSeries object
        self.series = series

        if metadata is not None:
            self.metadata = metadata

