from typing import NoReturn
from timeatlas import TimeSeries
from pandas import DataFrame
import fbprophet as fbp

from timeatlas.abstract import AbstractBaseModel


class Prophet(AbstractBaseModel):

    def __init__(self):
        super().__init__()

    def fit(self, series) -> NoReturn:
        super().fit(series)
        df = self.__prepare_series_for_prophet(self)
        m = fbp.Prophet()
        m.fit(df)

    def predict(self) -> Any:
        # TODO Continue here with future dataframe creation with Prophet make_future_dataframe method
        #   based on a duration taken from a timedelta object
        #   (https://docs.python.org/3/library/datetime.html#datetime.timedelta)
        super().predict()

    @staticmethod
    def __prepare_series_for_prophet(series: TimeSeries):
        df = series.to_df()
        df["ds"] = df.index
        df = df.reset_index(drop=True)
        df = df.rename(columns={"values": "y"})
        return df
