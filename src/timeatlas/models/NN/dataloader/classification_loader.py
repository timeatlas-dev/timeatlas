from torch.utils.data import Dataset
import torch
from timeatlas import TimeSeriesDataset


class TimeSeriesClassificationDataset(Dataset):
    """
    A DataLoader for the classification of complete TimeSeries, where X: TimeSeries and y: label of the TimeSeries
    """

    def __init__(self, data: TimeSeriesDataset):
        self.data = [torch.tensor(d.series) for d in data]
        self.labels = torch.tensor([ts.label for ts in data])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.labels[item]
