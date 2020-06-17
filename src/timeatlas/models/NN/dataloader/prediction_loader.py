from .base_loader import BaseDataset

from timeatlas import TimeSeriesDataset
from timeatlas.models.NN.util import chunkify


class TimeSeriesPredictionDataset(BaseDataset):
    """
    A DataLoader for the classification of complete TimeSeries, where X: TimeSeries and y: label of the TimeSeries
    """

    def __init__(self, data: TimeSeriesDataset, n: int or None):
        self.data, self.labels = chunkify(arr=data.data, seq_len=n)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.labels[item]
