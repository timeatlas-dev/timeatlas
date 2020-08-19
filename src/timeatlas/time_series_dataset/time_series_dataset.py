from typing import List, Any, NoReturn, Tuple

from pandas import DataFrame
import random

from timeatlas.time_series import TimeSeries
from timeatlas.utils import ensure_dir, to_pickle

from timeatlas.abstract import (
    AbstractAnalysis,
    AbstractOutputText,
    AbstractOutputPickle,
    AbstractProcessing
)


class TimeSeriesDataset(AbstractAnalysis, AbstractProcessing, AbstractOutputText, AbstractOutputPickle):
    """ Defines a set of time series

    A TimeSeriesDataset represent a set of TimeSeries
    objects.

    """

    def __init__(self, data: List[TimeSeries] = None):
        if data is None:
            self.data = []
        else:
            if isinstance(data, list):
                self.data = data
            elif isinstance(data, TimeSeries):
                self.data = [data]
            else:
                raise ValueError(f'data has to be TimeSeries or List[TimeSeries], got {type(data)}')

    def __getitem__(self, item: int) -> 'TimeSeries':
        return self.data[item]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return (ts for ts in self.data)

    # Methods
    # =======

    def add(self, time_series: TimeSeries):
        self.data.append(time_series)

    def remove(self, index):
        del self.data[index]

    def len(self):
        return len(self.data)

    def percent(self, percent: float, seed: int = None, indices: bool = False) -> Any:
        """

        returns a subset of the TimeSeriesDataset with randomly chosen percentage elements without replacement.

        Args:
            percent: percentage of elements returned
            seed: seed for random generator
            indices: if True returns the indices of the selection

        Returns: TimeSeriesDataset (optional: indices of selection)

        """

        # setting the seed if None no seed will be set automatically
        random.seed(seed)
        n = round(len(self.data) * percent)

        if n <= 0:
            raise ValueError(f'set percentage to small resulting selection is <= 0')

        if indices:
            return self.random(n=n, indices=indices)
        else:
            return self.random(n=n)

    def select(self, selection: List[int], indices: bool = False) -> Any:
        """

        select elements from the TimeSeriesDataset with a list of indices.

        Args:
            selection: list of indices
            indices: if True the selection is returned

        Returns: TimeSeriesDataset (optional: indices of selection)

        """
        if indices:
            return selection, TimeSeriesDataset([self.data[i] for i in selection])
        else:
            return TimeSeriesDataset([self.data[i] for i in selection])

    def random(self, n: int, seed: int = None, indices: bool = False) -> Any:
        """

        returns a subset of the TimeSeriesDataset with randomly chosen n elements without replacement.

        Args:
            n: number of elements returned
            seed: seed for random generator
            indices: if True returns the indices of the selection

        Returns: TimeSeriesDataset (optional: indices of selection)

        """

        # setting the seed if None no seed will be set automatically
        random.seed(seed)

        if indices:
            inds, data = zip(*random.sample(population=list(enumerate(self.data)), k=n))
            return list(inds), TimeSeriesDataset(list(data))
        else:
            TimeSeriesDataset(random.sample(population=self.data, k=n))

    # =============================================
    # Analysis
    # =============================================

    def describe(self) -> DataFrame:
        """
        Describe a TimeSeries with the describe function from Pandas

        Returns:
            TODO Define return type
        """
        pass

    def compute_resolution(self):
        """
        Create the series of delta T between each timestamp of a TimeSeries

        Returns:
            TODO Should it returns a Series of TimeDelta?
        """
        pass

    def compute_duration(self) -> Any:
        """
        Compute the duration of a TimeSeries by giving you its start and ending
        timestamps in a Tuple

        Returns:
            TODO Add stricter return type once found
        """
        pass

    def compute_correlations(self) -> Any:
        """
        Compute the correlations between all time series present in a
        TimeSeriesDataset

        Returns:
            TODO Add stricter return type once found
        """
        pass

    def duration(self):
        pass

    def frequency(self):
        pass

    def resolution(self):
        pass

    # =============================================
    # Processing
    # =============================================

    def resample(self, by: str) -> Any:
        """
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
        - upsampling
        - downsampling
        """
        pass

    def interpolate(self, method: str) -> Any:
        """
        "Intelligent" interpolation in function of the data unit etc.
        """
        pass

    def normalize(self, method: str) -> Any:
        """
        Normalize a dataset
        """
        pass

    def synchronize(self):
        """
        Synchronize the timestamps so that they are the same for all time
        series in the TimeSeriesDataset
        """
        pass

    def unify(self):
        """
        Put all time series in a matrix iff they all have the same length
        """
        pass

    # =============================================
    # IO
    # =============================================

    # Outputs

    def to_text(self, path: str) -> NoReturn:
        ensure_dir(path)
        for i, ts in enumerate(self.data):
            ts_path = "{}/{}".format(path, i)
            ensure_dir(ts_path)
            ts.to_text(ts_path)

    def to_pickle(self, path: str) -> NoReturn:
        to_pickle(self, path)
