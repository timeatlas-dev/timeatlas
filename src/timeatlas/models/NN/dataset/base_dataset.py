from torch.utils.data import Dataset, DataLoader, random_split

from typing import List


class BaseDataset(Dataset):
    """
    Base Class for the PyTorch DataSets
    """

    def __init__(self, data=None):
        self.data = data

        self.min = None
        self.max = None
        self.mu = None
        self.sigma = None

    def min_max_norm(self):
        """

        Normalization of the Dataset between 0 and 1

        x_norm_ji = (x_ji - min) / (max -min)

        """
        self.max, self.min = self.data.max(), self.data.min()
        self.data = (self.data - self.min) / (self.max - self.min)

    def z_score_norm(self):
        """

        Normalization according to the standard scoring

        x_norm_ij = (x_ji - mean) / std

        """
        self.mu, self.sigma = self.data.mean(), self.data.std()
        self.data = (self.data - self.mu) / self.sigma
