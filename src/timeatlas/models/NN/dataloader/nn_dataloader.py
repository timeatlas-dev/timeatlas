from torch.utils.data import Dataset, DataLoader

from timeatlas import TimeSeriesDataset


class NNTimeSeriesDataset(Dataset):

    def __init__(self, data: TimeSeriesDataset, labels=None):
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass
