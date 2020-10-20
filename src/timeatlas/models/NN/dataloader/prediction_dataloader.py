try:
    from torch.utils.data import Subset
except ModuleNotFoundError:
    raise ModuleNotFoundError("Pytorch not found. Install with pip install torch")

from .base_dataloader import BaseDataLoader


class PredictionDataLoader(BaseDataLoader):
    def __init__(self, dataset, *args, **kwargs):
        super(PredictionDataLoader, self).__init__(dataset=dataset, *args, **kwargs)

    def split(self, train: float, test: float or None = None, validation: float or None = None):
        """Split Dataset into train, test and validation

        Splitting a dataset into train, test and validation by percentages.

        Args:
            train: percentage of train split
            test: percentage of test split
            validation: percentage of validation split

        Returns: Subset of the DataSet

        """

        idx = list(range(len(self.dataset)))
        n = len(self.dataset)

        if validation is None:

            n_train = int(n * train)

            train_idx = idx[:n_train]
            test_idx = idx[n_train:]

            train_set = Subset(self.dataset, train_idx)
            test_set = Subset(self.dataset, test_idx)

            return train_set, test_set

        else:
            if test is None:
                test = 1 - train - validation

            n_train = int(n * train)
            n_test = int(n * test)

            train_idx = idx[:n_train]
            val_idx = idx[n_train:(n_train + n_test)]
            test_idx = idx[(n_train + n_test):]

            train_set = Subset(self.dataset, train_idx)
            val_set = Subset(self.dataset, val_idx)
            test_set = Subset(self.dataset, test_idx)

            return train_set, val_set, test_set
