from torch.utils.data import Dataset
import torch
from timeatlas import TimeSeriesDataset

class BaseDataset(Dataset):

    def min_max_norm(self):
        raise NotImplementedError

    def z_score_norm(self):
        raise NotImplementedError

    def sigmoid_norm(self):
        raise NotImplementedError
