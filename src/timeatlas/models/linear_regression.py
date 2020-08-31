from typing import NoReturn, Union, Tuple
from numpy import ndarray
from pandas import Series
from sklearn import linear_model

from timeatlas.time_series import TimeSeries
from timeatlas.abstract import AbstractBaseModel
from timeatlas.config.constants import TIME_SERIES_VALUES


class LinearRegression(AbstractBaseModel):

    def __init__(self):
        super().__init__()
        self.model = linear_model.LinearRegression()

    def fit(self, series: TimeSeries) -> NoReturn:
        """
        Fit a linear regression model given a time series

        Args:
            series: the TimeSeries to fit
        """
        super().fit(series)
        X_train, y_train = self.__prepare_series_for_sklearn(self.X_train)
        self.model.fit(X_train, y_train)

    def predict(self, horizon: Union[str, TimeSeries], freq: str = None) \
            -> TimeSeries:
        """
        Predict a TimeSeries given a horizon

        Args:
            horizon: str as in https://pandas.pydata.org/pandas-docs/stable/user_guide/timedeltas.html
            freq: frequency in DateOffset string https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects

        Returns:
            TimeSeries
        """
        super().predict(horizon)
        if isinstance(horizon, str):
            future, index = self.make_future_arrays(horizon)
        elif isinstance(horizon, TimeSeries):
            future, y_train = self.__prepare_series_for_sklearn(horizon)
            index = horizon.series.index
        forecast = self.model.predict(future)
        return TimeSeries(Series(data=forecast, index=index))

    @staticmethod
    def __prepare_series_for_sklearn(ts: TimeSeries)\
            -> Tuple[ndarray, ndarray]:
        """
        Prepare a TimeSeries object so that it can be given to a Scikit Learn
        model for prediction or other task

        Args:
            ts: the TimeSeries to prepare

        Returns:
            A Tuple with two elements:
                - a Numpy ndarray with shape (n, 1)
                - a Numpy ndarray with shape (n,)
        """
        X_train = ts.series.index.factorize()[0].reshape(-1, 1)
        y_train = ts.series[TIME_SERIES_VALUES].to_numpy()
        return X_train, y_train

    def make_future_arrays(self, horizon: str, freq: str = None)\
            -> Tuple[ndarray, ndarray]:
        """
        Create the arrays needed for the prediction

        Args:
            horizon: str as in https://pandas.pydata.org/pandas-docs/stable/user_guide/timedeltas.html
            freq: frequency in DateOffset string https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects

        Returns:
            A Tuple with two elements:
                - a Numpy ndarray with shape (n, 1)
                - a Numpy ndarray with shape (n,)

        """
        index = self.make_future_index(horizon, freq)
        X_train = self.__prepare_series_for_sklearn(self.X_train)
        X_test = index.factorize()[0].reshape(-1, 1) + len(X_train)
        return X_test, index

