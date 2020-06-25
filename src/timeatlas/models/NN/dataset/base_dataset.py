from torch.utils.data import Dataset, DataLoader, random_split

from typing import List


class BaseDataset(Dataset):
    """
    Base Class for the PyTorch DataSets
    """

    def __init__(self):
        pass