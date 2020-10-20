try:
    from torch.utils.data import Dataset
except ModuleNotFoundError:
    raise ModuleNotFoundError("Pytorch not found. Install with pip install torch")


class BaseDataset(Dataset):
    """
    Base Class for the PyTorch DataSets
    """

    def __init__(self, tsd):
        self.tsd = tsd
