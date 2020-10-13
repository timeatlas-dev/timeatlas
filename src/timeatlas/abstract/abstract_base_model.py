from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, NoReturn, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from timeatlas.time_series import TimeSeries
    from timeatlas.time_series_dataset import TimeSeriesDataset

from pandas import Timedelta, date_range, infer_freq


class AbstractBaseModel(ABC):
    """ Abstract class to define methods to implement
    for a Model class inspired by Scikit Learn API.
    """

    def __init__(self):
        self._is_fitted = False
        self.X_train = None
        self.type = None

    @abstractmethod
    def fit(self, series: Union["TimeSeries", "TimeSeriesDataset"]) \
            -> NoReturn:
        """ Fit a model """
        self._is_fitted = True
        self.X_train = series

    @abstractmethod
    def predict(self, horizon: Union[str, "TimeSeries", "TimeSeriesDataset"]) \
            -> Any:
        """ Usage of the model to predict values """
        if self._is_fitted is False:
            raise Exception('fit() must be called before predict()')

    def make_future_index(self, horizon: str, freq: str = None):
        """ Creates a DatetimeIndex from the last timestamp given
        in the training set X_train for a given horizon.

        Args:
            horizon: str as in https://pandas.pydata.org/pandas-docs/stable/user_guide/timedeltas.html
            freq: frequency in DateOffset string https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects

        Returns:
            DatetimeIndex
        """
        start = self.X_train.series.index[-1]
        end = self.X_train.series.index[-1] + Timedelta(horizon)
        if freq is not None:
            freq = freq
        else:
            freq = infer_freq(self.X_train.series.index)
        return date_range(start=start, end=end, freq=freq)[0:]
