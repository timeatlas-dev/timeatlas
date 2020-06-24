from torch.utils.data import Dataset, DataLoader, random_split

from typing import List


class BaseDataset(Dataset):
    """
    Base Class for the PyTorch DataSets
    """

    def __init__(self, data=None):
        self.data = data

    def min_max_norm(self):
        """

        Normalization of the Dataset between 0 and 1

        x_norm_ji = (x_ji - min) / (max -min)

        Returns: normalized data

        """
        xmax, xmin = self.data.max(), self.data.min()
        data_norm = (self.data - xmin) / (xmax - xmin)
        return data_norm

    def z_score_norm(self):
        """

        Normalization according to the standard scoring

        x_norm_ij = (x_ji - mean) / std

        Returns: normalized data

        """
        mu, sigma = self.data.mean(), self.data.std()
        data_norm = (self.data - mu) / sigma
        return data_norm
