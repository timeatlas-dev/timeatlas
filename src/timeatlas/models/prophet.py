from typing import NoReturn, Union, Optional
from pandas import DataFrame
import fbprophet as fbp

from timeatlas.abstract import AbstractBaseModel
from timeatlas.config.constants import (
    TIME_SERIES_VALUES,
    TIME_SERIES_CI_UPPER,
    TIME_SERIES_CI_LOWER
)
from timeatlas.time_series import TimeSeries
from timeatlas.time_series_dataset import TimeSeriesDataset

from timeatlas.plots.time_series import prediction


class Prophet(AbstractBaseModel):

    def __init__(self):
        super().__init__()
        self.model = fbp.Prophet()

    def fit(self, ts: Union[TimeSeries, TimeSeriesDataset],
            y: Optional[int] = None) -> NoReturn:
        """
        Fit a Prophet model to a time series. If given a TimeSeriesDataset, the
        optional argument y must be given to indicate on which component the
        model should be fitted to.

        Args:
            ts: TimeSeries or TimeSeriesDataset to fit
            y: Optional int of the component index in a TimeSeriesDataset

        Returns:
            NoReturn
        """
        super().fit(ts)

        if isinstance(ts, TimeSeries):
            df = self.__prepare_time_series_for_prophet(self.X_train)
        elif isinstance(ts, TimeSeriesDataset):
            df = self.__prepare_time_series_dataset_for_prophet(self.X_train, y)

            # TODO Continue here by adding the add_regressors() in a loop!

        else:
            ValueError('The fit method  accepts only TimeSeries or '
                       'TimeSeriesDataset as argument')

        self.model.fit(df)

    def predict(self, horizon: Union[str, TimeSeries], freq: str = None) \
            -> TimeSeries:
        super().predict(horizon)

        if isinstance(horizon, str):
            future = self.make_future_dataframe(horizon, freq)
            metadata = None
        elif isinstance(horizon, TimeSeries):
            future = self.__prepare_time_series_for_prophet(horizon.empty())
            metadata = horizon.metadata

        forecast = self.model.predict(future)
        forecast.rename(columns={"yhat": TIME_SERIES_VALUES,
                                 "yhat_lower": TIME_SERIES_CI_LOWER,
                                 "yhat_upper": TIME_SERIES_CI_UPPER},
                        inplace=True)
        df = forecast[[TIME_SERIES_VALUES,
                       TIME_SERIES_CI_LOWER,
                       TIME_SERIES_CI_UPPER]]
        df.index = forecast["ds"]

        # Register the prediction plot
        ts = TimeSeries(df, metadata)
        ts.register_plotting_function(lambda x: prediction(x))

        return ts


    @staticmethod
    def __prepare_time_series_for_prophet(ts: TimeSeries):
        df = ts.to_df().copy()
        df["ds"] = df.index
        df = df.reset_index(drop=True)
        df = df.rename(columns={"values": "y"})
        return df

    @staticmethod
    def __prepare_time_series_dataset_for_prophet(tsd: TimeSeriesDataset,
                                                  y: int):
        df = tsd.to_df().copy()
        df["ds"] = df.index
        df = df.reset_index(drop=True)
        df = df.rename(columns={y: "y"})
        return df

    def make_future_dataframe(self, horizon: str, freq: str = None):
        index = self.make_future_index(horizon, freq)
        df = DataFrame(data=index.to_series(), columns=["ds"])
        df = df.reset_index(drop=True)
        return df
