from typing import NoReturn, Union, Optional

from pandas import DataFrame
try:
    import fbprophet as fbp
except ModuleNotFoundError:
    raise ModuleNotFoundError("Could not find fbprophet. Install with: pip install fbprophet")

from timeatlas.abstract import AbstractBaseModel
from timeatlas.config.constants import (
    TIME_SERIES_VALUES,
    TIME_SERIES_CI_UPPER,
    TIME_SERIES_CI_LOWER,
    MODEL_TYPE_UNIVARIATE,
    MODEL_TYPE_MULTIVARIATE
)
from timeatlas.time_series import TimeSeries
from timeatlas.time_series_dataset import TimeSeriesDataset


class Prophet(AbstractBaseModel):

    def __init__(self):
        super().__init__()
        self.model = fbp.Prophet()
        self.y = None

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

        # Prepare the model
        if isinstance(ts, TimeSeries):
            self.type = MODEL_TYPE_UNIVARIATE
            df = self.__prepare_time_series_for_prophet(self.X_train)

        elif isinstance(ts, TimeSeriesDataset):
            assert y is not None, "For multivariate prediction, the y " \
                                  "argument must be given."
            self.type = MODEL_TYPE_MULTIVARIATE
            self.y = y
            df = self.__prepare_time_series_dataset_for_prophet(
                self.X_train,
                self.y
            )

            # Add all components except y as extra regressor
            regressors = df.columns.to_list()
            regressors.remove("y")
            regressors.remove("ds")
            for r in regressors:
                self.model.add_regressor(r)
        else:
            ValueError('The fit method  accepts only TimeSeries or '
                       'TimeSeriesDataset as argument')

        self.model.fit(df)

    def predict(self, horizon: Union[str, TimeSeries, TimeSeriesDataset],
            freq: str = None) \
            -> TimeSeries:
        super().predict(horizon)

        # Prepare the data
        if self.type == MODEL_TYPE_UNIVARIATE:
            if isinstance(horizon, str):
                future = self.make_future_dataframe(horizon, freq)
                metadata = None
            elif isinstance(horizon, TimeSeries):
                future = self.__prepare_time_series_for_prophet(horizon.empty())
                metadata = horizon.metadata

        elif self.type == MODEL_TYPE_MULTIVARIATE:
            if isinstance(horizon, TimeSeriesDataset):
                horizon[:, self.y] = horizon[:, self.y].empty()
                future = self.__prepare_time_series_dataset_for_prophet(
                    horizon, self.y)
                metadata = horizon[:, self.y].data[self.y].metadata

        else:
            ValueError("horizon argument type isn't recognized")

        # Predict
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
        return ts

    @staticmethod
    def __prepare_time_series_for_prophet(ts: TimeSeries):
        df = ts.to_df().copy()
        df["ds"] = df.index
        df = df.reset_index(drop=True)
        df = df.rename(columns={"values": "y"})
        df.columns = df.columns.astype(str)
        return df

    @staticmethod
    def __prepare_time_series_dataset_for_prophet(tsd: TimeSeriesDataset,
            y: int):
        df = tsd.to_df().copy()
        df["ds"] = df.index
        df = df.reset_index(drop=True)
        df = df.rename(columns={y: "y"})
        df.columns = df.columns.astype(str)
        return df

    def make_future_dataframe(self, horizon: str, freq: str = None):
        index = self.make_future_index(horizon, freq)
        df = DataFrame(data=index.to_series(), columns=["ds"])
        df = df.reset_index(drop=True)
        return df
