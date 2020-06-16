from torch.utils.data import Dataset, DataLoader

from timeatlas import TimeSeriesDataset
from timeatlas.models.NN.util import chunkify


class TimeSeriesPredictionDataset(Dataset):
    """
    A dataloader for the classification of complete Timeseries, where X: TimeSeries and y: label of the TimeSeries
    """

    def __init__(self, data: TimeSeriesDataset, n: int or None):
        self.data, self.labels = chunkify(arr=data.data, seq_len=n)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.labels[item]
