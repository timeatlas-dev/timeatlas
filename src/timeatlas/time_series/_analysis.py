from typing import Tuple, TYPE_CHECKING
from datetime import datetime

from timeatlas.abstract.abstract_analysis import AbstractAnalysis

if TYPE_CHECKING:
    from .time_series import TimeSeries


class Analysis(AbstractAnalysis):

    def describe(self):
        """
        Describe a TimeSeries with the describe function from Pandas

        Args:
            time_series:

        Returns:
            TODO
        """
        raise NotImplementedError
        pass

    def compute_resolution(self) -> 'TimeSeries':
        """
        Create the series of delta T between each timestamp of a TimeSeries
        TODO Should it returns a Series of TimeDelta?

        Args:
            time_series: the resolution TimeSeries

        Returns:
            TODO
        """
        raise NotImplementedError
        pass

    def compute_duration(self) -> Tuple[datetime, datetime]:
        """
        Compute the duration of a TimeSeries by giving you its start and ending
        timestamps in a Tuple

        Args:
            time_series: TimeSeries

        Returns:
            TODO
        """
        raise NotImplementedError
        pass
