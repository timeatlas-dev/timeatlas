import random
from collections import defaultdict
from copy import deepcopy
from typing import List, Any, NoReturn, Tuple, Union, Optional
from warnings import warn

import numpy as np
from pandas import DataFrame, Timestamp, Timedelta, concat
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import DateOffset

from timeatlas.abstract import (
    AbstractBaseTimeSeries,
    AbstractOutputText,
    AbstractOutputPickle
)
from timeatlas.time_series import TimeSeries
from timeatlas.plots.time_series_dataset import line_plot
from timeatlas.utils import ensure_dir, to_pickle


class TimeSeriesDataset(List,
                        AbstractBaseTimeSeries,
                        AbstractOutputText,
                        AbstractOutputPickle):
    """ Defines a set of time series

    A TimeSeriesDataset represent a set of TimeSeries
    objects.

    """

    def __init__(self, data: List[TimeSeries] = None):
        super().__init__()
        if data is not None:
            for i in data:
                if isinstance(i, TimeSeries):
                    self.append(i)
                else:
                    raise TypeError(f"All elements of data must be a "
                                    f"TimeSeries; got {type(i)}")

    def __repr__(self):
        description = self.describe()
        return description.__repr__()

    def __setitem__(self, item, value) -> NoReturn:
        if isinstance(value, TimeSeries):
            self[item] = value
        else:
            raise ValueError("Value argument must be TimeSeries")

    # ==========================================================================
    # Methods
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

    def append(self, item: TimeSeries) -> None:
        """Append a TimeSeries to TimeSeriesDataset

        Appending an item to the TimeSeriesDataset, if item is of type TimeSeries.

        Args:
            item: item to append

        Returns: None

        """
        if not isinstance(item, TimeSeries):
            raise ValueError(f"Only items of type TimeSeries can be appended to TimeSeriesDataset; got {type(item)}")
        else:
            super().append(item)

    def plot(self) -> Any:
        """Plot a TimeSeriesDataset

        Returns:
            plotly.graph_objects.Figure
        """
        return line_plot(self)

    def copy(self, deep=True) -> 'TimeSeriesDataset':
        """Copy a list

        Copy the TSD to either a deep or shallow copy of itself

        Args:
            deep: if True, creates a deep copy else a shallow one

        Returns: (deep) copy of TimeSeriesDataset

        """
        return deepcopy(self) if deep else super().copy()

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
        arr = [self[i].split_at(timestamp) for i in range(len(self))]
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
        for ts in self:
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
        """Fill all values in each TimeSeries from a TimeSeriesDataset.

        If given a unique value, all values will be broadcast. If given an
        array of the length of the TimeSeriesDataset, it will replace all
        values.

        Args:
            value: Any values that you want to fill the TimeSeriesDataset with

        Returns:
            TimeSeries
        """
        return TimeSeriesDataset([ts.fill(value) for ts in self])

    def empty(self) -> 'TimeSeriesDataset':
        """Empty the values in each TimeSeries from a TimeSeriesDataset.

        This method is equivalent to filling all values of each TimeSeries in a
        TimeSeriesDataset with NaNs.

        Returns:
            TimeSeriesDataset
        """
        return self.fill(np.nan)

    def pad(self, limit: Union[int, str, Timestamp], side: Optional[str] = None,
            value: Any = np.NaN) -> 'TimeSeriesDataset':
        """
        Pad a TimeSeriesDataset until a given limit

        Args:
            limit: int, str or Pandas Timestamp
                if int, it will pad the side given in the side arguments by n
                elements.

            side: Optional[str]
                side to which the TimeSeries will be padded. This arg can have
                two value: "before" and "after" depending where the padding is
                needed.

                This arg is needed only in case the limit is given in int.

            value: Any values

        Returns:
            TimeSeriesDataset
        """
        return TimeSeriesDataset([ts.pad(limit=limit, side=side, value=value)
                                  for ts in self])

    def trim(self, side: str = "both") -> 'TimeSeriesDataset':
        """Remove NaNs from a TimeSeries start, end or both

        Args:
            side:
                the side where to remove the NaNs. Valid values are either
                "start", "end" or "both". Default to "both"

        Returns:
            TimeSeries
        """
        return TimeSeriesDataset([ts.trim(side) for ts in self])

    def merge(self, tsd: 'TimeSeriesDataset') -> 'TimeSeriesDataset':
        """Merge two TimeSeriesDataset by the index of the TimeSeries

        This methods merges the TimeSeries from the TimeSeriesDataset (TSD) in
        argument with self based on the indexes of each one of the TSDs.

        Args:
            tsd: the TimeSeriesDataset to merge with self

        Returns:
            TimeSeriesDataset
        """
        arr = []
        for i, ts in enumerate(self):
            merged_ts = ts.merge(tsd[i])
            arr.append(merged_ts)
        return TimeSeriesDataset(arr)

    def merge_by_label(self, tsd: 'TimeSeriesDataset') -> 'TimeSeriesDataset':
        """Merge two TimeSeriesDatasets by the label of the TimeSeries
        in the TimeSeriesDatasets

        Args:
            tsd: TimeSeriesDataset to be merged with self

        Returns:
            TimeSeriesDataset
        """

        def list_duplicates(seq: list) -> list:
            """Get label duplicates

            Getting a dict of with key=ts.label and value=index in TSD

            Args:
                seq: list of labels

            Returns: dict of duplicate indices

            """
            tally = defaultdict(list)
            for i, item in enumerate(seq):
                tally[item].append(i)

            duplicates_items = ((key, locs) for key, locs in tally.items())

            duplicates = []
            for dup in sorted(duplicates_items):
                duplicates.append(dup)

            return duplicates

        def merge_duplicates(tsd: TimeSeriesDataset, duplicates: list) \
                -> TimeSeriesDataset:
            """Merging based on duplicates

            Merging the TimeSeriesDataset based on the duplicate list.

            Args:
                tsd: TimeSeriesDataset to be merged
                duplicates: list of tuples with (label,duplicate indices)

            Returns: merged TimeSeriesDataset

            """
            arr = []
            for label, inds in duplicates:
                # We merge everything into the first occurrence of the label
                # -> tsd[inds[0]]
                base_ts = tsd[inds[0]]
                if len(inds) > 1:
                    for ind in inds[1:]:
                        base_ts = base_ts.merge(tsd[ind])
                    arr.append(base_ts)
                else:
                    arr.append(base_ts)

            return TimeSeriesDataset(arr)

        # First : Create new TSD and add all elements
        merged_tsd = self.copy()
        for ts in tsd:
            merged_tsd.append(ts)

        # Second : Get list of tuples (label, duplicate indices)
        tsd_labels = [ts.label for ts in merged_tsd]
        duplicates = list_duplicates(seq=tsd_labels)

        # Third : Merge TSD, where TS has same labels.
        merged_tsd = merge_duplicates(tsd=merged_tsd, duplicates=duplicates)

        return merged_tsd

    # TimeSeriesDataset
    # -----------------

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
                population=list(enumerate(self)), k=n))
            return list(inds), TimeSeriesDataset(list(data))
        else:
            TimeSeriesDataset(random.sample(population=self, k=n))

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
        n = round(len(self) * percent)

        # Workaround: If percentage too small we select at least 1
        if n <= 0:
            warn(
                f'set percentage to small resulting selection is <= 0\n Using n=1.')
            n = 1
        if indices:
            return self.select_components_randomly(n=n, indices=indices)
        else:
            return self.select_components_randomly(n=n)

    def shuffle(self, inplace: bool = False) -> 'TimeSeriesDataset':
        """Randomizing the order of the TS in the TSD

        Randomizing the order of the TS in TSDs and returning a new TSD.

        Args:
            inplace: randomizing inplace or creating new object. (Default: False)

        Returns: shuffled TimeSeriesDataset

        """

        if inplace:
            random.shuffle(self)
        else:
            new_tsd = self.copy()
            random.shuffle(new_tsd)
            return new_tsd

    # ==========================================================================
    # Processing
    # ==========================================================================

    # TimeSeries
    # ----------

    def apply(self, func, tsd: 'TimeSeriesDataset' = None) \
            -> 'TimeSeriesDataset':
        # TODO (See GitHub issue 56)
        raise NotImplementedError

    def resample(self, freq: Union[str, TimeSeries],
            method: Optional[str] = None) \
            -> 'TimeSeriesDataset':
        """Convert the TimeSeries in a TimeSeriesDataset to a specified
        frequency. Optionally provide filling method to pad/backfill missing
        values.

        Args:
            freq: str or TimeSeries.
                The new time difference between two adjacent entries in the
                returned TimeSeries.

                If a TimeSeries is given, the freq in self will become
                the same as in the given TimeSeries.

                If a string is given, three options are available:
                    * 'lowest' : to sync the TimeSeriesDataset to the lowest frequency
                    * 'highest' : to sync the TimeSeriesDataset to the highest frequency
                    * A DateOffset alias e.g.: 15min, H, etc.


            method: {'backfill'/'bfill', 'pad'/'ffill'}, default None
                Method to use for filling holes in reindexed Series (note this
                does not fill NaNs that already were present):

                * 'pad'/'ffil': propagate last valid observation forward to next
                  valid
                * 'backfill'/'ffill': use next valid observation to fill

        Returns:
            TimeSeriesDataset
        """
        # Acting differently depending on freq arg type and value
        if isinstance(freq, TimeSeries):
            target_freq = freq.frequency()
        elif freq == "lowest":
            frequencies = [to_offset(ts.frequency()) for ts in self]
            target_freq = max(frequencies)
        elif freq == "highest":
            frequencies = [to_offset(ts.frequency()) for ts in self]
            target_freq = min(frequencies)
        elif isinstance(to_offset(freq), DateOffset):
            target_freq = freq
        else:
            raise ValueError("freq argument isn't valid")
        # Resample and return
        return TimeSeriesDataset(
            [ts.resample(target_freq, method) for ts in self])

    def group_by(self, freq: str, method: Optional[str] = "mean") \
            -> 'TimeSeriesDataset':
        """Groups values by a frequency for each TimeSeries in a
        TimeSeriesDataset.

        This method is quite similar to resample with the difference that it
        gives the guaranty that the timestamps are full values.
        e.g. 2019-01-01 08:00:00.

        Resample could make values spaced by 1 min but
        every x sec e.g. [2019-01-01 08:00:33, 2019-01-01 08:01:33],
        which isn't convenient for further index merging operations.

        The function has different aggregations methods taken from Pandas
        groupby aggregations[1]. By default, it'll take the mean of the
        defined freq bucket.

        [1] https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#aggregation

        Args:
            freq: string offset alias of a frequency
            method: string of the Pandas aggregation function.

        Returns:
            TimeSeriesDataset
        """
        return TimeSeriesDataset([ts.group_by(freq, method) for ts in self])

    def interpolate(self, *args, **kwargs) -> 'TimeSeriesDataset':
        """Wrapper around the Pandas interpolate() method.

        See https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.interpolate.html
        for reference
        """
        return TimeSeriesDataset(
            [ts.interpolate(*args, **kwargs) for ts in self])

    def normalize(self, method: str) -> 'TimeSeriesDataset':
        """Normalize the TimeSeries in a TimeSeriesDataset with a given method

        Args:
            method: str
            * 'minmax' for a min max normalization
            * 'zscore' for Z score normalization

        Returns:
            TimeSeriesDataset
        """
        return TimeSeriesDataset([ts.normalize(method) for ts in self])

    def round(self, decimals: int) -> 'TimeSeriesDataset':
        """Round the values of every TimeSeries in the TimeSeriesDataset with a
        defined number of digits

        Args:
            decimals: int defining the number of digits after the comma

        Returns:
            TimeSeriesDataset
        """
        return TimeSeriesDataset([ts.round(decimals) for ts in self])

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
        return TimeSeriesDataset([ts.sort(*args, **kwargs) for ts in self])

    # TimeSeriesDataset
    # -----------------
    def regularize(self, side: str = "[]", fill=np.nan):
        """
        Regularize a TimeSeriesDataset so that all starting and ending
        timestamps are similar. This method keeps the frequency of every
        TimeSeries the same.

        fill : user defined value, mean, median, etc.
        method : ][, [[, ]], []

        return
            TSD
        """
        # 1. find the boundaries of all TSs
        all_boundaries = self.boundaries()

        # 2. find earliest/latest values
        starts = np.array(all_boundaries).T[0]
        earliest_start = min(starts)
        latest_start = max(starts)
        ends = np.array(all_boundaries).T[1]
        earliest_end = min(ends)
        latest_end = max(ends)

        # 3. pad all TSs accordingly
        if side == "[]":
            tsd = self.pad(earliest_start).pad(latest_end)
        elif side == "[[":
            tsd = TimeSeriesDataset([ts[:earliest_end]
                                     for ts in self.pad(earliest_start)])
        elif side == "]]":
            tsd = TimeSeriesDataset([ts[latest_start:]
                                     for ts in self.pad(latest_end)])
        elif side == "][":
            tsd = TimeSeriesDataset([ts[latest_start:earliest_end]
                                     for ts in self])
        else:
            raise ValueError("method argument is not recognized")

        return tsd

    # ==========================================================================
    # Analysis
    # ==========================================================================

    # Basic Statistics
    # ----------------

    def min(self) -> List[Any]:
        return [ts.min() for ts in self]

    def max(self) -> List[Any]:
        return [ts.max() for ts in self]

    def mean(self) -> List[Any]:
        return [ts.mean() for ts in self]

    def median(self) -> List[Any]:
        return [ts.median() for ts in self]

    def kurtosis(self) -> List[Any]:
        return [ts.kurtosis() for ts in self]

    def skewness(self) -> List[Any]:
        return [ts.skewness() for ts in self]

    def describe(self) -> DataFrame:
        """Describe a TimeSeriesDataset with the describe function from Pandas

        TODO Define what describe should do on TimeSeriesDataset (see issue 56)

        Returns:
            TODO Define return type
        """
        min = self.min()
        max = self.max()
        mean = self.mean()
        median = self.median()
        kurtosis = self.kurtosis()
        skewness = self.skewness()

        return DataFrame.from_dict({'minimum': min,
                                    'maximum': max,
                                    'mean': mean,
                                    'median': median,
                                    'kurtosis': kurtosis,
                                    'skewness': skewness})

    # Time Series Statistics
    # ----------------------

    def start(self) -> List[Timestamp]:
        """Get the first Timestamp of a all components of
        a TimeSeriesDataset

        Returns:
            List of Pandas Timestamp
        """
        return [ts.start() for ts in self]

    def end(self) -> List[Timestamp]:
        """Get the last Timestamp of a all components of
        a TimeSeriesDataset

        Returns:
            List of Pandas Timestamp
        """
        return [ts.end() for ts in self]

    def boundaries(self) -> List[Tuple[Timestamp, Timestamp]]:
        """Get the tuple with the TimeSeries first and last index for all
        components in the TimeSeriesDataset

        Returns:
            List of Tuple of Pandas Timestamps
        """
        return [ts.boundaries() for ts in self]

    def frequency(self) -> List[Optional[str]]:
        """Get the frequency of a each TimeSeries in a TimeSeriesDataset

        Returns:
            List of str or None
                - str of the frequency according to the Pandas Offset Aliases
                - None if no discernible frequency
        """
        return [ts.frequency() for ts in self]

    def resolution(self) -> 'TimeSeriesDataset':
        """Compute the time difference between each timestamp for all TimeSeries
        in a TimeSeriesDataset

       Returns:
           TimeSeriesDataset
       """
        return TimeSeriesDataset(
            [ts.resolution() for ts in self])

    def duration(self) -> List[Timedelta]:
        """Get the duration for all TimeSeries in a TimeSeriesDataset

        Returns:
            a List of Pandas Timedelta
        """
        return [ts.duration() for ts in self]

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
        for i, ts in enumerate(self):
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

    def to_df(self) -> DataFrame:
        """Converts a TimeSeriesDataset to a Pandas DataFrame

        The indexes of all the TimeSeries in the TimeSeriesDataset will get
        merged. That means that you are the only responsible if the merge
        induces the addition of many NaNs in your data. Therefore, it's better
        to use this method if you are sure that your TimeSeries share a common
        frequency as well as start and end.

        Returns:
            DataFrame
        """

        # get the result ready
        columns = []
        res = self[0].to_df()
        for i, col in enumerate(res.columns):
            columns.append(f"{col}_{i}")
        # merge the rest in the result
        rest = self[1:]
        for i, ts in enumerate(rest):
            res = res.merge(ts.series, how="outer",
                            left_index=True, right_index=True)
            cols = ts.series.columns
            for col in cols:
                columns.append(f"{col}_{i+1}")
        res.columns = columns
        return res

    def to_array(self) -> np.ndarray:
        """TimeSeriesData to NumpyArray [n x len(tsd)], where n is number of TimeSeries in dataset
        # TODO Should output a warning if the ts have different length
            (issue 56)

        Returns: numpy.array of shape (n x len(tsd))
        """
        return np.array([ts.to_array() for ts in self], dtype=object)

    def to_darts(self):
        """Convert a TimeSeriesDataset to Darts TimeSeries

        Returns:
            Darts TimeSeries object
        """
        # TODO issue 56
        raise NotImplementedError
