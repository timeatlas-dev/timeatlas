from torch.utils.data import Dataset

from sklearn.preprocessing import normalize

import numpy as np


class BaseDataset(Dataset):

    def __init__(self, data=None):
        self.data = data

    def min_max_norm(self, min: int = 0, max: int = 1):
        xmax, xmin = self.data.max(), self.data.min()
        data_norm = (self.data - xmin) / (xmax - xmin)
        return data_norm

    def z_score_norm(self):
        mu, sigma = self.data.mean(), self.data.std()
        data_norm = (self.data - mu) / sigma
        return data_norm
