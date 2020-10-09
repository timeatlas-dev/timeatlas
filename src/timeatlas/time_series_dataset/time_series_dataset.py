from typing import List, Any, NoReturn, Tuple, Union, Optional

import numpy as np
from pandas import DataFrame, Timestamp, Timedelta
import random

from timeatlas.time_series import TimeSeries
from timeatlas.utils import ensure_dir, to_pickle
from timeatlas.abstract import (
    AbstractBaseTimeSeries,
    AbstractOutputText,
    AbstractOutputPickle
)


class TimeSeriesDataset(AbstractBaseTimeSeries,
                        AbstractOutputText,
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

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return (ts for ts in self.data)

    def __repr__(self):
        # TODO
        return "{}".format(len(self.data))

    def __getitem__(self, item: int) -> 'TimeSeries':
        return self.data[item]

    # ==========================================================================
    # Method
    # ==========================================================================

    # TimeSeries
    # ----------

    @staticmethod
    def create(length: int, start: str, end: str,
            freq: Union[str, 'TimeSeries'] = None) \
            -> 'TimeSeriesDataset':
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

    def plot(self, *args, **kwargs) -> Any:
        # TODO
        raise NotImplementedError

    def split_at(self, timestamp: Union[str, Timestamp]) \
            -> Tuple['TimeSeriesDataset', 'TimeSeriesDataset']:
        """Split a TimeSeriesDataset at a defined point and include the
        splitting point in both as in [start,...,at] and [at,...,end].

        Args:
            timestamp: str or Timestamp where to the TimeSeriesDataset will be
                split (e.g. "2019-12-31 00:00:00")

        Returns:
            a Tuple of TimeSeriesDataset ([start,...,at] and [at,...,end])

        """
        arr = [self.data[i].split_at(timestamp) for i in range(len(self))]
        first_split = TimeSeriesDataset([split_ts[0] for split_ts in arr])
        second_split = TimeSeriesDataset([split_ts[1] for split_ts in arr])
        return first_split, second_split

    def split_in_chunks(self, n: int) -> List['TimeSeriesDataset']:
        """The TimeSeries in the TimeSeriesDataset are cut into chunks
        of length n

        Args:
            n: length of the individual chunks

        Returns: List of TimeSeriesDatasets containing the chunks
        """
        # Split all TS in the TSD in chunks and store them in an array
        chunkified_ts_arr = []
        for ts in self.data:
            chunkified_ts = ts.split_in_chunks(n)
            chunkified_ts_arr.append(chunkified_ts)
        # Create n TSDs containing each the ith chunks of all TS. Then add each
        # TSD in an array
        chunkified_tsd_arr = []
        for ith_chunk in range(n):
            tsd_data = []
            for chunkified_ts in chunkified_ts_arr:
                tsd_data.append(chunkified_ts[ith_chunk])
            tsd = TimeSeriesDataset(tsd_data)
            chunkified_tsd_arr.append(tsd)
        return chunkified_tsd_arr

    def fill(self, value: Any) -> 'TimeSeriesDataset':
        """Fill a TimeSeriesDataset with a value. If given a unique value, all
        values will be broadcasted. If given an array of the length of the
        TimeSeriesDataset, it will replace all values.

        Args:
            value: Any values that you want to fill the TimeSeriesDataset with

        Returns:
            TimeSeries
        """
        return TimeSeriesDataset([ts.fill(value) for ts in self.data])

    def empty(self) -> 'TimeSeriesDataset':
        """Empty the TimeSeriesDataset (fill all values with NaNs)

        Returns:
            TimeSeriesDataset
        """
        return self.fill(np.nan)

    def trim(self, side: str = "both") -> 'TimeSeriesDataset':
        """Remove NaNs from a TimeSeries start, end or both

        Args:
            side:
                the side where to remove the NaNs. Valid values are either
                "start", "end" or "both". Default to "both"

        Returns:
            TimeSeries
        """
        return TimeSeriesDataset([ts.trim(side) for ts in self.data])

    def merge_by_index(self, tsd: 'TimeSeriesDataset') -> 'TimeSeriesDataset':
        """Merge two TimeSeriesDataset by the index of the TimeSeries

        This methods merges the TimeSeries from the TimeSeriesDataset (TSD) in
        argument with self based on the indexes of each one of the TSDs.

        Args:
            tsd: the TimeSeriesDataset to merge with self

        Returns:
            TimeSeries
        """
        arr = []
        for i, ts in enumerate(self.data):
            merged_ts = ts.merge(tsd[i])
            arr.append(merged_ts)
        return TimeSeriesDataset(arr)

    # TimeSeriesDataset
    # -----------------

    def add_component(self, time_series: TimeSeries) -> 'TimeSeriesDataset':
        """
        Add a time series to the time series dataset

        Args:
            time_series: the TimeSeries object to add
        """
        return TimeSeriesDataset(self.data.append(time_series))

    def remove_component(self, index) -> 'TimeSeriesDataset':
        """
        Remove a time series from the time series dataset by its index

        Args:
            index: int of the time series to remove
        """
        del self.data[index]
        return TimeSeriesDataset(self.data)

    def len(self) -> int:
        """
        Return the number of time series in the time series dataset

        Returns:
            int of the number of time series
        """
        return len(self.data)

    def select_components_by_index(self, selection: List[int],
            indices: bool = False) -> Any:
        """Select elements from the TimeSeriesDataset with a list of indices.

        Args:
            selection: list of indices
            indices: if True the selection is returned

        Returns: TimeSeriesDataset (optional: indices of selection)

        """
        if indices:
            return selection, TimeSeriesDataset([self.data[i] for i in selection])
        else:
            return TimeSeriesDataset([self.data[i] for i in selection])

    def select_components_randomly(self, n: int, seed: int = None,
            indices: bool = False) -> Any:
        """Returns a subset of the TimeSeriesDataset with randomly chosen n
        elements without replacement.

        Args:
            n: number of elements returned
            seed: seed for random generator
            indices: if True returns the indices of the selection

        Returns: TimeSeriesDataset (optional: indices of selection)

        """
        # setting the seed if None no seed will be set automatically
        random.seed(seed)
        if indices:
            inds, data = zip(*random.sample(
                population=list(enumerate(self.data)), k=n))
            return list(inds), TimeSeriesDataset(list(data))
        else:
            TimeSeriesDataset(random.sample(population=self.data, k=n))

    def select_components_by_percentage(self, percent: float, seed: int = None,
            indices: bool = False) -> Any:
        """Returns a subset of the TimeSeriesDataset with randomly chosen
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
            raise ValueError(f'set percentage to small resulting selection is '
                             f'<= 0')
        if indices:
            return self.select_components_randomly(n=n, indices=indices)
        else:
            return self.select_components_randomly(n=n)

    def shuffle(self, inplace: bool = False) -> 'TimeSeriesDataset':
        """

        Randomizing the order of the TS in the TSD

        Args:
            inplace: randomizing inplace or creating new object. (Default: False)

        Returns: if inplace = True shuffling self.data else return new TSD with randomized self.data

        """

        if inplace:
            random.shuffle(self.data)
        else:
            new_data = self.data
            random.shuffle(new_data)
            return TimeSeriesDataset(data=new_data)

    def merge(self, ts: 'TimeSeries') -> 'TimeSeries':
        """

        PLACEHOLDER

        Args:
            ts:

        Returns:

        """
        raise NotImplementedError

    # ==========================================================================
    # Processing
    # ==========================================================================

    # TimeSeries
    # ----------

    def apply(self, func, tsd: 'TimeSeriesDataset' = None) \
            -> 'TimeSeriesDataset':
        # TODO (See GitHub issue 56)
        raise NotImplementedError

    def resample(self, freq: str, method: Optional[str] = None) \
            -> 'TimeSeriesDataset':
        """Convert the TimeSeries in a TimeSeriesDataset to a specified
        frequency. Optionally provide filling method to pad/backfill missing
        values.

        Args:
            freq: string
                The new time difference between two adjacent entries in the
                returned TimeSeries. A DateOffset alias is expected.

            method: {'backfill'/'bfill', 'pad'/'ffill'}, default None
                Method to use for filling holes in reindexed Series (note this
                does not fill NaNs that already were present):

                * 'pad'/'ffil': propagate last valid observation forward to next
                  valid
                * 'backfill'/'ffill': use next valid observation to fill

        Returns:
            TimeSeriesDataset
        """
        return \
            TimeSeriesDataset([ts.resample(freq, method) for ts in self.data])

    def interpolate(self, *args, **kwargs) -> 'TimeSeriesDataset':
        """Wrapper around the Pandas interpolate() method.

        See https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.interpolate.html
        for reference
        """
        return TimeSeriesDataset(
            [ts.interpolate(*args, **kwargs) for ts in self.data])

    def normalize(self, method: str) -> 'TimeSeriesDataset':
        """Normalize the TimeSeries in a TimeSeriesDataset with a given method

        Args:
            method: str
            * 'minmax' for a min max normalization
            * 'zscore' for Z score normalization

        Returns:
            TimeSeriesDataset
        """
        return TimeSeriesDataset(
            [ts.normalize(method) for ts in self.data])

    def round(self, decimals: int) -> 'TimeSeriesDataset':
        """Round the values of every TimeSeries in the TimeSeriesDataset with a
        defined number of digits

        Args:
            decimals: int defining the number of digits after the comma

        Returns:
            TimeSeriesDataset
        """
        return TimeSeriesDataset([ts.round(decimals) for ts in self.data])

    def sort(self, *args, **kwargs) -> 'TimeSeriesDataset':
        """Sort the TimeSeries of a TimeSeriesDataset by time stamps

        Basically, it's a wrapper around df.sort_index()
        see: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_index.html

        Args:
            args: the positional arguments
            kwargs: the keyword arguments

        Returns:
            TimeSeriesDataset
        """
        return TimeSeriesDataset([ts.sort(*args, **kwargs) for ts in self.data])

    # TimeSeriesDataset
    # -----------------

    def synchronize(self):
        """
        Synchronize the timestamps so that they are the same for all time
        series in the TimeSeriesDataset
        """
        # TODO
        raise NotImplementedError

    # ==========================================================================
    # Analysis
    # ==========================================================================

    # Basic Statistics
    # ----------------

    def min(self) -> List[Any]:
        return [ts.min() for ts in self.data]

    def max(self) -> List[Any]:
        return [ts.max() for ts in self.data]

    def mean(self) -> List[Any]:
        return [ts.mean() for ts in self.data]

    def median(self) -> List[Any]:
        return [ts.median() for ts in self.data]

    def kurtosis(self) -> List[Any]:
        return [ts.kurtosis() for ts in self.data]

    def skewness(self) -> List[Any]:
        return [ts.skewness() for ts in self.data]

    def describe(self) -> DataFrame:
        """Describe a TimeSeriesDataset with the describe function from Pandas

        TODO Define what describe should do on TimeSeriesDataset (see issue 56)

        Returns:
            TODO Define return type
        """
        raise NotImplementedError

    # Time Series Statistics
    # ----------------------

    def start(self) -> List[Timestamp]:
        """Get the first Timestamp of a all components of
        a TimeSeriesDataset

        Returns:
            List of Pandas Timestamp
        """
        return [ts.start() for ts in self.data]

    def end(self) -> List[Timestamp]:
        """Get the last Timestamp of a all components of
        a TimeSeriesDataset

        Returns:
            List of Pandas Timestamp
        """
        return [ts.end() for ts in self.data]

    def boundaries(self) -> List[Tuple[Timestamp, Timestamp]]:
        """Get the tuple with the TimeSeries first and last index for all
        components in the TimeSeriesDataset

        Returns:
            List of Tuple of Pandas Timestamps
        """
        return [ts.boundaries() for ts in self.data]

    def frequency(self) -> List[Optional[str]]:
        """Get the frequency of a each TimeSeries in a TimeSeriesDataset

        Returns:
            List of str or None
                - str of the frequency according to the Pandas Offset Aliases
                - None if no discernible frequency
        """
        return [ts.frequency() for ts in self.data]

    def resolution(self) -> 'TimeSeriesDataset':
        """Compute the time difference between each timestamp for all TimeSeries
        in a TimeSeriesDataset

       Returns:
           TimeSeriesDataset
       """
        return TimeSeriesDataset(
            [ts.resolution() for ts in self.data])

    def duration(self) -> List[Timedelta]:
        """Get the duration for all TimeSeries in a TimeSeriesDataset

        Returns:
            a List of Pandas Timedelta
        """
        return [ts.duration() for ts in self.data]

    # =============================================
    # IO
    # =============================================

    # Outputs

    def to_text(self, path: str) -> NoReturn:
        """Export a TimeSeriesDataset to text format

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
        """Creating a pickle out of the TimeSeriesDataset

        Args:
            path: Path, where the TimeSeriesDataset will be saved

        Returns: NoReturn
        """
        to_pickle(self, path)

    def to_dataframe(self) -> DataFrame:
        # TODO issue 56
        raise NotImplementedError

    def to_array(self) -> np.ndarray:
        """TimeSeriesData to NumpyArray [n x len(tsd)], where n is number of TimeSeries in dataset
        # TODO Should output a warning if the ts have different length
            (issue 56)

        Returns: numpy.array of shape (n x len(tsd))
        """
        return np.array([ts.to_array() for ts in self.data], dtype=object)

    def to_darts(self):
        """Convert a TimeSeriesDataset to Darts TimeSeries

        Returns:
            Darts TimeSeries object
        """
        # TODO issue 56
        raise NotImplementedError
