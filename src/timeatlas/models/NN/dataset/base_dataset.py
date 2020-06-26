from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """
    Base Class for the PyTorch DataSets
    """

    def __init__(self, tsd):
        self.tsd = tsd
