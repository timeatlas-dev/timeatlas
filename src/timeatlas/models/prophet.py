from typing import NoReturn, Union
from pandas import DataFrame
import fbprophet as fbp

from timeatlas.abstract import AbstractBaseModel
from timeatlas.config.constants import TIME_SERIES_VALUES
from timeatlas.time_series import TimeSeries


class Prophet(AbstractBaseModel):

    def __init__(self):
        super().__init__()
        self.model = fbp.Prophet()

    def fit(self, series) -> NoReturn:
        super().fit(series)
        df = self.__prepare_series_for_prophet(self.X_train)
        self.model.fit(df)

    def predict(self, horizon: Union[str, TimeSeries], freq: str = None) -> TimeSeries:
        super().predict(horizon)
        if isinstance(horizon, str):
            future = self.make_future_dataframe(horizon, freq)
        elif isinstance(horizon, TimeSeries):
            future = self.__prepare_series_for_prophet(horizon.empty())
        forecast = self.model.predict(future)
        forecast.rename(columns={"yhat": TIME_SERIES_VALUES,
                                 "yhat_lower": "ci_lower",
                                 "yhat_upper": "ci_upper"},
                        inplace=True)
        df = forecast[[TIME_SERIES_VALUES, 'ci_lower', 'ci_upper']]
        df.index = forecast["ds"]
        return TimeSeries(df)

    @staticmethod
    def __prepare_series_for_prophet(series: TimeSeries):
        df = series.to_df()
        df["ds"] = df.index
        df = df.reset_index(drop=True)
        df = df.rename(columns={"values": "y"})
        return df

    def make_future_dataframe(self, horizon: str, freq: str = None):
        index = self.make_future_index(horizon, freq)
        df = DataFrame(data=index.to_series(), columns=["ds"])
        df = df.reset_index(drop=True)
        return df
