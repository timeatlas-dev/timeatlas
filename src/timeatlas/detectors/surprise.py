from typing import List, Callable, Tuple, Union

from timeatlas.time_series import TimeSeries
from timeatlas.abstract import AbstractBaseDetector, AbstractBaseModel
from timeatlas.processors import Scaler, miscellaneous
from timeatlas.config.constants import TIME_SERIES_VALUES


class Surprise(AbstractBaseDetector):
    """
    A detector that checks on unexpected data signals
    """

    def __init__(self, model: AbstractBaseModel, error: Callable):
        super().__init__()

        # Params
        self.model = model
        self.error = error
        self.normalizer = None
        self._compute_thresholds_params = None

        # Intermediate results
        self.truth = None
        self.prediction = None
        self.surprise = None
        self.thresholds = None

        # Object state
        self._is_fitted = False

    def normalize(self, method: str) -> 'Surprise':
        """Normalizing Data

        Set the normalizer for the data

        Args:
            method: string that sets the normalizing type

        Returns:

        """
        if method == "minmax":
            self.normalizer = Scaler.minmax
        elif method == "zscore":
            self.normalizer = Scaler.zscore
        return self

    def alerts(self, method: str = "quantile", thresholds: Union[float, List] = [0.85, 0.95]) -> 'Surprise':
        """Set alerts

        Threshold alerts given,

        Args:
            method: string that sets the method
            thresholds: threshold of given alert

        Returns:
            Surprise

        """
        self._compute_thresholds_params = (method, thresholds)
        return self

    def fit(self, ts: TimeSeries) -> 'Surprise':
        """Fit the Surprise Detector

        Args:
            ts: time-series to check for surprises

        Returns:
            Surprise

        """
        # Set the default alerts if not custom
        if self._compute_thresholds_params is None:
            self.alerts()

        # Set the truth in the Surprise anomaly detector
        self.truth = ts

        # Predict and compute the actual surprise
        self.prediction, self.surprise = self.__compute_surprise(
            self.model, ts, self.error)

        # Normalize if desired
        if self.normalizer is not None:
            self.surprise = self.normalizer(self.surprise)

        # Compute the thresholds according to the given truth
        self.thresholds = self.__compute_thresholds(
            self._compute_thresholds_params[0],
            self._compute_thresholds_params[1],
            self.surprise
        )

        self._is_fitted = True
        return self

    def detect(self, ts: TimeSeries) -> TimeSeries:
        """Detect Surprises

        Args:
            ts: time-series to find surprises in

        Returns:
            TimeSeries

        """
        prediction, surprise = self.__compute_surprise(self.model, ts,
                                                       self.error)

        if self.normalizer is not None:
            surprise = self.normalizer(surprise)

        alerts = miscellaneous.ceil(surprise, self.thresholds)
        return alerts

    @staticmethod
    def __compute_surprise(m: AbstractBaseModel, ts: TimeSeries,
            error_func: Callable) \
            -> Tuple['TimeSeries', 'TimeSeries']:
        prediction = m.predict(ts)
        surprise = ts.apply(error_func, prediction)
        return prediction, surprise

    @staticmethod
    def __compute_thresholds(method: str, thresholds: List,
            ts: TimeSeries = None):
        res = []
        if method == "quantile":
            for k, v in enumerate(thresholds):
                res.append(ts.series[TIME_SERIES_VALUES].quantile(q=v))
        elif method == "threshold":
            res = thresholds
        else:
            raise Exception("method {} isn't implemented".format(method))
        return res
