from .base_dataset import BaseDataset
from timeatlas.time_series_dataset import TimeSeriesDataset
import numpy as np


class TimeSeriesClassificationDataset(BaseDataset):
    """
    A DataLoader for the classification of complete TimeSeries, where X: TimeSeries and y: label of the TimeSeries
    """

    def __init__(self, timeseriesdataset: TimeSeriesDataset):
        super(TimeSeriesClassificationDataset, self).__init__(tsd=timeseriesdataset)
        self.data = np.array([ts.series for ts in timeseriesdataset])
        self.labels = [ts.label for ts in timeseriesdataset]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.labels[item]
