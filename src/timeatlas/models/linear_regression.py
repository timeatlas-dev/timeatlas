from typing import NoReturn, Union
from pandas import Series
from sklearn import linear_model

from timeatlas.time_series import TimeSeries

from timeatlas.abstract import AbstractBaseModel


class LinearRegression(AbstractBaseModel):

    def __init__(self):
        super().__init__()
        self.model = linear_model.LinearRegression()

    def fit(self, series) -> NoReturn:
        super().fit(series)
        X_train, y_train = self.__prepare_series_for_sklearn(self.X_train)
        self.model.fit(X_train, y_train)

    def predict(self, horizon: Union[str, TimeSeries], freq: str = None) -> NoReturn:
        super().predict(horizon)
        if isinstance(horizon, str):
            future, index = self.make_future_array(horizon)
        elif isinstance(horizon, TimeSeries):
            future, y_train = self.__prepare_series_for_sklearn(horizon)
            index = horizon.series.index
        forecast = self.model.predict(future)
        return TimeSeries(Series(data=forecast, index=index))

    @staticmethod
    def __prepare_series_for_sklearn(ts: TimeSeries):
        X_train = ts.series.index.factorize()[0].reshape(-1, 1)
        y_train = ts.series.to_numpy()
        return X_train, y_train

    def make_future_array(self, horizon, freq: str = None):
        index = self.make_future_index(horizon, freq)
        X_train = self.__prepare_series_for_sklearn(self.X_train)
        X_test = index.factorize()[0].reshape(-1, 1) + len(X_train)
        return X_test, index