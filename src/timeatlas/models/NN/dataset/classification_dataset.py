from .base_dataset import BaseDataset
from timeatlas import TimeSeriesDataset
import numpy as np


class TimeSeriesClassificationDataset(BaseDataset):
    """
    A DataLoader for the classification of complete TimeSeries, where X: TimeSeries and y: label of the TimeSeries
    """

    def __init__(self, data: TimeSeriesDataset):
        super(TimeSeriesClassificationDataset, self).__init__(data=data)
        self.data = np.array([d.series for d in data])
        self.labels = [ts.label for ts in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.labels[item]
