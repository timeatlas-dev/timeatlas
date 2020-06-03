from typing import NoReturn
from timeatlas import TimeSeries
from pandas import DataFrame, Timedelta
import fbprophet as fbp

from timeatlas.abstract import AbstractBaseModel


class Prophet(AbstractBaseModel):

    def __init__(self):
        super().__init__()
        self.m = fbp.Prophet()

    def fit(self, series) -> NoReturn:
        super().fit(series)
        df = self.__prepare_series_for_prophet(self.X_train)
        self.m.fit(df)

    def predict(self, horizon: str, freq: str = None) -> NoReturn:
        super().predict(horizon)
        future = self.make_future_dataframe(horizon, freq)
        forecast = self.m.predict(future)
        return forecast

    @staticmethod
    def __prepare_series_for_prophet(series: TimeSeries):
        df = series.to_df()
        df["ds"] = df.index
        df = df.reset_index(drop=True)
        df = df.rename(columns={"values": "y"})
        return df

    def make_future_dataframe(self, horizon, freq: str = None):
        index = self.make_future_index(horizon, freq)
        df = DataFrame(data=index.to_series(), columns=["ds"])
        df = df.reset_index(drop=True)
        return df
