from typing import NoReturn, Union
from pandas import DataFrame
import fbprophet as fbp

from timeatlas.abstract import AbstractBaseModel
from timeatlas.config.constants import TIME_SERIES_VALUES
from timeatlas.time_series import TimeSeries
from timeatlas.plots.time_series import prediction


class Prophet(AbstractBaseModel):

    def __init__(self):
        super().__init__()
        self.model = fbp.Prophet()

    def fit(self, ts: TimeSeries) -> NoReturn:
        super().fit(ts)
        df = self.__prepare_series_for_prophet(self.X_train)
        self.model.fit(df)

    def predict(self, horizon: Union[str, TimeSeries], freq: str = None) -> TimeSeries:
        super().predict(horizon)

        if isinstance(horizon, str):
            future = self.make_future_dataframe(horizon, freq)
            metadata = None
        elif isinstance(horizon, TimeSeries):
            future = self.__prepare_series_for_prophet(horizon.empty())
            metadata = horizon.metadata

        forecast = self.model.predict(future)
        forecast.rename(columns={"yhat": TIME_SERIES_VALUES,
                                 "yhat_lower": "ci_lower",
                                 "yhat_upper": "ci_upper"},
                        inplace=True)
        df = forecast[[TIME_SERIES_VALUES, 'ci_lower', 'ci_upper']]
        df.index = forecast["ds"]

        # Register the prediction plot
        ts = TimeSeries(df, metadata)
        ts.register_plotting_function(lambda x: prediction(x))

        return ts


    @staticmethod
    def __prepare_series_for_prophet(ts: TimeSeries):
        df = ts.to_df().copy()
        df["ds"] = df.index
        df = df.reset_index(drop=True)
        df = df.rename(columns={"values": "y"})
        return df

    def make_future_dataframe(self, horizon: str, freq: str = None):
        index = self.make_future_index(horizon, freq)
        df = DataFrame(data=index.to_series(), columns=["ds"])
        df = df.reset_index(drop=True)
        return df
