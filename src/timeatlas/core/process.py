from src.timeatlas.data_structures import Dataset, Object

class Process:

    def resample(self, dataset: Dataset, by: str):
        """
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
        - upsampling
        - downsampling
        """
        raise NotImplementedError

    def interpolate(self, dataset: Dataset):
        """
        Intelligent interpolation in function of the data unit etc.
        """
        raise NotImplementedError

    def normalize(self, dataset: Dataset, method: str):
        """
        Normalize a dataset
        """
        raise NotImplementedError

    def reselect(self, dataset: Dataset, column: str):
        raise NotImplementedError

    def unify(self, dataset: Dataset):
        raise NotImplementedError

    def augment(self, dataset: Dataset, object: Object):
        raise NotImplementedError

    def group_by(self, dataset: Dataset):
        raise NotImplementedError


