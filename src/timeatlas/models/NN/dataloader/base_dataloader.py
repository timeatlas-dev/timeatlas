from torch.utils.data import DataLoader, random_split


class BaseDataLoader(DataLoader):
    def __init__(self, dataset, *args, **kwargs):
        super(BaseDataLoader, self).__init__(dataset=dataset, *args, **kwargs)

    def split(self, train: float, test: float or None = None, validation: float or None = None):
        """

        Randomly split a dataset into non-overlapping new datasets of given percentages.

        Args:
            train: percentage in train set
            test: percentage in test set (if None given: test =  1 - train)
            validation: percentage in validation set (if None: validation = 1 - train - test)

        Returns: list of DataLoader-objects for each set

        """

        if validation is None:

            train_length = int(len(self.dataset) * train)
            test_length = int(len(self.dataset) - train_length)

            return [BaseDataLoader(item) for item in random_split(self.dataset, lengths=[train_length, test_length])]

        else:
            if test is None:
                test = 1 - train - validation

            train_length = int(len(self.dataset) * train)
            test_length = int(len(self.dataset) * test)
            val_length = int(len(self.dataset) - train_length - test_length)

            return [BaseDataLoader(item) for item in
                    random_split(self.dataset, lengths=[train_length, test_length, val_length])]
