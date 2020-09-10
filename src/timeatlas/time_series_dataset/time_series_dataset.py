from typing import List, Any, NoReturn, Tuple, Union

from pandas import DataFrame
from numpy import array, ndarray
import random

from timeatlas.time_series import TimeSeries
from timeatlas.metadata import Metadata
from timeatlas.utils import ensure_dir, to_pickle

from timeatlas.abstract import (
    AbstractBaseTimeSeries,
    AbstractOutputText,
    AbstractOutputPickle
)


class TimeSeriesDataset(AbstractBaseTimeSeries, AbstractOutputText,
                        AbstractOutputPickle):
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

    def __repr__(self):
        return self.data.__repr__()

    # ==========================================================================
    # Methods
    # ==========================================================================

    def add(self, time_series: TimeSeries):
        """
        Add a time series to the time series dataset

        Args:
            time_series: the TimeSeries object to add
        """
        self.data.append(time_series)

    def remove(self, index):
        """
        Remove a time series from the time series dataset by its index

        Args:
            index: int of the time series to remove
        """
        del self.data[index]

    def len(self):
        """
        Return the number of time series in the time series dataset

        Returns:
            int of the number of time series
        """
        return len(self.data)

    @staticmethod
    def create(length: int, start: str, end: str, freq: Union[str, 'TimeSeries'] = None):
        """
        Create an empty TimeSeriesDataset object with a defined index and period

        Args:
            length: int representing the number of TimeSeries to include in the
                TimeSeriesDataset
            start: str of the start of the DatetimeIndex
                (as in Pandas.date_range())
            end: the end of the DatetimeIndex (as in Pandas.date_range())
            freq: the optional frequency it can be a str or a TimeSeries
                (to copy its frequency)

        Returns:
            TimeSeriesDataset
        """
        # Check length parameter
        assert length >= 1, 'Length must be >= 1'
        data = []
        ts = TimeSeries.create(start, end, freq)
        for i in range(length):
            data.append(ts)
        return TimeSeriesDataset(data)

    def percent(self, percent: float, seed: int = None, indices: bool = False) -> Any:
        """

        returns a subset of the TimeSeriesDataset with randomly chosen
        percentage elements without replacement.

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

    def chunkify(self, n: int) -> List['TimeSeriesDataset']:
        """

        The TimeSeries in the TimeSeriesDataset are cut into chunks of length n

        Args:
            n: length of the individual chunks

        Returns: List of TimeSeriesDatasets containing the chunks

        """

        tsd_chunks = []

        for ts in self.data:
            ts_chunks = ts.chunkify(n=n)
            tsd_chunks.append(TimeSeriesDataset(ts_chunks))

        return tsd_chunks

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
        """

        Args:
            path: Path, where the TimeSeriesDataset will be saved in

        Returns: NoReturn

        """
        ensure_dir(path)
        for i, ts in enumerate(self.data):
            ts_path = "{}/{}".format(path, i)
            ensure_dir(ts_path)
            ts.to_text(ts_path)

    def to_pickle(self, path: str) -> NoReturn:
        """

        Creating a pickle out of the TimeSeriesDataset

        Args:
            path: Path, where the TimeSeriesDataset will be saved

        Returns: NoReturn

        """
        to_pickle(self, path)

    def to_array(self) -> ndarray:
        """

        TimeSeriesData to NumpyArray [n x len(tsd)], where n is number of TimeSeries in dataset

        Returns: numpy.array of shape (n x len(tsd))

        """

        return array([ts.to_array() for ts in self.data], dtype=object)
