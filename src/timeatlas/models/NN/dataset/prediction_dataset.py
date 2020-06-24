from .base_dataset import BaseDataset

from timeatlas import TimeSeriesDataset
from timeatlas.models.NN.util import chunkify


class TimeSeriesPredictionDataset(BaseDataset):
    """
    A DataLoader for the prediction of a TimeSeries n next steps, where
    X: TimeSeries n previous steps
    y:  next step of the TimeSeries
    """

    def __init__(self, data: TimeSeriesDataset, n: int or None):
        """

        Args:
            data: TimeSeriesDataset
            n: number of previous steps
        """
        super(TimeSeriesPredictionDataset, self).__init__()
        self.data, self.labels = chunkify(tsd=data.data, seq_len=n)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.labels[item]
