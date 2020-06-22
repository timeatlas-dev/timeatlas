from typing import List

from timeatlas import TimeSeries
from timeatlas.abstract import AbstractBaseDetector, AbstractBaseModel
from timeatlas.processing import scalers, miscellaneous


class Surprise(AbstractBaseDetector):

    def __init__(self):
        super().__init__()
        self.truth = None
        self.prediction = None
        self.surprise = None
        self.thresholds = None
        self.alerts = None

    def compute(self, model: AbstractBaseModel, ts: TimeSeries, error_func):
        self.prediction = model.predict(ts)
        self.truth = ts
        self.surprise = self.truth.apply(error_func, self.prediction)
        return self

    def normalize(self, method: str):
        if method == "minmax":
            self.surprise = scalers.minmax(self.surprise)
        elif method == "zscore":
            self.surprise = scalers.zscore(self.surprise)
        return self

    def set_alerts(self, method: str, thresholds: List = [0.85, 0.95]):
        if method == "quantile":
            quantile_thresholds = []
            for k, v in enumerate(thresholds):
                quantile_thresholds.append(self.surprise.series.quantile(q=v))
            self.thresholds = quantile_thresholds
        return self

    def detect(self) -> TimeSeries:
        self.alerts = miscellaneous.ceil(self.surprise, self.thresholds)
        return self
