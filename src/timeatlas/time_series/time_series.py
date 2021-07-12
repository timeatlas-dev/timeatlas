from typing import NoReturn, Tuple, Any, Union, Optional, List
from copy import deepcopy, copy
from warnings import warn

from darts import TimeSeries as DartsTimeSeries
import numpy as np
from pandas import DataFrame, date_range, infer_freq, Series, DatetimeIndex, \
    Timestamp, Timedelta, concat

from timeatlas.abstract import (
    AbstractBaseTimeSeries,
    AbstractOutputText,
    AbstractOutputPickle
)
from timeatlas.config.constants import (
    COMPONENT_VALUES,
    TIME_SERIES_FILENAME,
    TIME_SERIES_EXT,
    METADATA_FILENAME,
    METADATA_EXT
)

from timeatlas.metadata import Metadata
from timeatlas.processors.scaler import Scaler
from timeatlas.plots.time_series import line_plot
from timeatlas.utils import ensure_dir, to_pickle
from timeatlas.time_series.component import Component
from timeatlas.time_series.component_handler import ComponentHandler


class TimeSeries(AbstractBaseTimeSeries, AbstractOutputText, AbstractOutputPickle):
    """
     A TimeSeries object is a series of time indexed values.
    """
    """Defines a time series

    A TimeSeries object is a series of time indexed values.

    Attributes:
        series: An optional Pandas DataFrame
        metadata: An optional Dict storing metadata about this TimeSeries
    """

    def __init__(self,
            data: DataFrame = None,
            handler: ComponentHandler = None):
        """Defines a time series

        A TimeSeries object is a series of time indexed values.

        Args:
            data: DataFrame containing the values and labels
            handler: ComponentHandler
        """
        if data is not None:

            # Perform preliminary checks
            # --------------------------

            assert isinstance(data, DataFrame), \
                'data must be of a DataFrame.'

            assert isinstance(data.index, DatetimeIndex), \
                'Values must be indexed with a DatetimeIndex.'

            assert len(data) >= 1, \
                'Values must have at least one values.'

            assert len(data.columns) >= 1, \
                "DataFrame must have at least one column."

            # Create the TimeSeries object
            # ----------------------------

            # Create the components handler
            if handler is None:
                self._handler = ComponentHandler()
                for col in data.columns:
                    component = Component(col)
                    self._handler.append(component)
            else:
                self._handler = handler

            # Rename the columns
            data.columns = self._handler.get_columns()

            # Store the data with certainty that values are sorted
            self._data = data.sort_index()

            # Add the freq if regular
            if len(data) >= 3:
                self._data.index.freq = infer_freq(self._data.index)

            # Create instance variables
            self.index = self._data.index  # index accessor
            self.values = self._data[
                self._handler.get_columns().to_list()
            ]

        else:

            # Create empty structures
            self._data = DataFrame()
            self._handler = ComponentHandler()

    def __repr__(self):
        return self._data.__repr__()

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return (v for i, v in self._data.iterrows())

    def __getitem__(self, item: Union[int, str, Timestamp,
                                      slice,
                                      List[int], List[str]]):

        # ts[0] -> select rows
        if isinstance(item, int):
            new_handler = self._handler
            new_data = self._data.iloc[[item]]

        # ts["0_foo"] -> select columns
        elif isinstance(item, str):
            new_handler = self._handler[item]
            new_data = self._data.loc[:, new_handler.get_columns()]

        # ts[my_timestamp] -> select rows
        elif isinstance(item, Timestamp):
            new_handler = self._handler
            new_data = self._data.loc[[item]]

        elif isinstance(item, slice):

            # ts[0:4] -> select rows
            if isinstance(item.start, int) or isinstance(item.stop, int):
                new_handler = self._handler
                new_data = self._data.iloc[item]

            # ts["2013":"2014"] -> select rows
            elif isinstance(item.start, str) or isinstance(item.stop, str):
                new_handler = self._handler
                new_data = self._data.loc[item]

            else:
                raise KeyError(f"rows can't be sliced with type {type(item)}")

        elif isinstance(item, list):

            # ts[[0,3,5]] -> select columns
            if all(isinstance(i, int) for i in item):
                new_handler = self._handler[item]
                new_data = self._data.iloc[:, item]

            # ts[["a",... ,"b"]] -> select columns
            elif all(isinstance(i, str) for i in item):
                new_handler = self._handler[item]
                new_data = self._data.loc[:, item]

            else:
                raise TypeError(f"TimeSeries can't be selected with list of "
                                f"type {type(item)}")

        else:
            raise TypeError(f"TimeSeries can't be selected with type "
                            f"{type(item)}")

        return TimeSeries(new_data, new_handler)

    # ==========================================================================
    # Methods
    # ==========================================================================

    @staticmethod
    def create(start: str, end: str,
            freq: Union[str, 'TimeSeries'] = None) \
            -> 'TimeSeries':
        """Creates an empty TimeSeries object with the period as index

        Args:
            start: str of the start of the DatetimeIndex (as in Pandas.date_range())
            end: the end of the DatetimeIndex (as in Pandas.date_range())
            freq: the optional frequency it can be a str or a TimeSeries (to copy its frequency)

        Returns:
            TimeSeries
        """
        if freq is not None:
            if isinstance(freq, TimeSeries):
                freq = infer_freq(freq._data.index)
            elif isinstance(freq, str):
                freq = freq
        data = DataFrame(columns=[COMPONENT_VALUES],
                         index=date_range(start, end, freq=freq))
        return TimeSeries(data)

    def stack(self, ts: 'TimeSeries'):
        """ Stack two TimeSeries together

        Create a unique TimeSeries from two TimeSeries so that the resulting
        TimeSeries has the component(s) from self and ts.

        Args:
            ts: the TimeSeries to stack

        Returns:
            TimeSeries

        """
        assert (self.index == ts.index).all(), \
            "Indexes aren't the same"
        new_data = concat([self._data, ts._data], axis=1)
        new_components = self._handler.copy()
        for c in ts._handler.components:
            new_components.append(c)
        return TimeSeries(new_data, new_components)

    def drop(self, key: Union[int, str]) -> 'TimeSeries':
        """ Drop a component of a TimeSeries by its index

        Args:
            key: int or str of the component to delete

        Returns:
            TimeSeries
        """
        # Given the current state of the TimeSeries, get the name of the columns
        # that will compose the new_data
        all_cols = self._handler.get_columns().to_list()
        if isinstance(key, int):
            cols_to_remove = self._handler.get_column_by_id(key).to_list()
        elif isinstance(key, str):
            cols_to_remove = self._handler.get_column_by_name(key).to_list()
        else:
            raise TypeError(f"key must be int or str, not {type(key)}")
        new_cols = self.__list_diff(all_cols, cols_to_remove)

        # select only the leftover data from self.data
        new_data = self._data.copy()
        new_data = new_data[new_cols]

        # drop the component to get rid off
        new_handler = self._handler.copy()
        del new_handler[key]

        return TimeSeries(new_data, new_handler)

    def plot(self, *args, **kwargs) -> Any:
        """Plot a TimeSeries

        Returns:
            plotly.graph_objects.Figure
        """
        return line_plot(self, *args, **kwargs)

    def copy(self, deep=False) -> 'TimeSeries':
        """Copy a TimeSeries

        Copy the TSD to either a deep or shallow copy of itself

        Args:
            deep: if True, creates a deep copy else a shallow one

        Returns: (deep) copy of TimeSeries

        """
        return deepcopy(self) if deep else copy(self)

    def split_at(self, timestamp: Union[str, Timestamp]) \
            -> Tuple['TimeSeries', 'TimeSeries']:
        """Split a TimeSeries at a defined point and include the splitting point
        in both as in [start,...,at] and [at,...,end].

        Args:
            timestamp: str or Timestamp where to the TimeSeries will be split
                            (e.g. "2019-12-31 00:00:00")

        Returns:
            a Tuple of TimeSeries ([start,...,at] and [at,...,end])

        """
        start = self._data.index[0]
        end = self._data.index[-1]
        first_split = self._data[start:timestamp].copy()
        second_split = self._data[timestamp:end].copy()
        before = TimeSeries(first_split, self._handler)
        after = TimeSeries(second_split, self._handler)
        return before, after

    def split_in_chunks(self, n: int) -> List['TimeSeries']:
        """Split a TimeSeries into chunks of length n

        When the number of element in the TimeSeries is not a multiple of n, the
        last chunk will have a length smaller than n.

        Args:
            n: length of the chunks

        Returns:
            List of TimeSeries
        """
        ts_chunks = [TimeSeries(data=v, handler=self._handler) for n, v in
                     self._data.groupby(np.arange(len(self._data)) // n)]
        return ts_chunks

    def sliding(self, size: int, step: int = 1) -> List['TimeSeries']:
        """

        Creates windows of the TimeSeries. If size > step the windows will be overlapping.

        Args:
            size: size of the window
            step: step size between windows

        Returns: List of TimeSeries

        """

        if size < step:
            warn(
                f"Windows size ({size}) is bigger than step size ({step}). The resulting data will jump over some values.")

        _rolling_data = [TimeSeries(v, handler=self._handler) for v in self._data.rolling(size) if len(v) == size]

        return _rolling_data[::step]

    def fill(self, value: Any) -> 'TimeSeries':
        """Fill a TimeSeries with values

        Fill a TimeSeries with a value. If given a unique value, all values
        will be broadcast. If given an array of the length of the TimeSeries,
        it will replace all values.

        Args:
            value: Any values that you want to fill the TimeSeries with

        Returns:
            TimeSeries
        """
        new_data = self.copy(deep=True)
        new_data._data[:] = value
        return new_data

    def empty(self) -> 'TimeSeries':
        """Empty the TimeSeries (fill all values with NaNs)

        Replace all values of a TimeSeries with NaNs

        Returns:
            TimeSeries
        """
        return self.fill(np.nan)

    def pad(self, limit: Union[int, str, Timestamp], side: Optional[str] = None, value: Any = np.NaN) -> 'TimeSeries':
        """Pad a TimeSeries until a given limit

        Padding a TimeSeries on left or right sides.

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
            TimeSeries
        """

        def create_pad(new_limit, ts, fill_val):
            """

            Local utility function to create padding time series from a Pandas
            Timestamp for a given TimeSeries

            Args:
                new_limit: Pandas Timestamp of the new limit to pad from/to
                ts: TimeSeries to pad
                fill_val: value to fill the TimeSeries with

            Returns:
                TimeSeries
            """
            if new_limit < ts.start():
                return ts.create(new_limit, ts.start(), freq=ts) \
                           .fill(fill_val)[:-1]
            elif new_limit > ts.end():
                return ts.create(ts.end(), new_limit, freq=ts) \
                           .fill(fill_val)[1:]
            if new_limit == ts.start() or new_limit == ts.end():
                return TimeSeries()
            else:
                raise ValueError("The given limit is included in the time "
                                 "series, padding is impossible")

        # Create padding TimeSeries from a given number of elements to pad with
        if isinstance(limit, int):
            # Add 1 to make the interval is too small
            target_limit = limit + 1

            if side == "before":
                index = date_range(end=self.start(), freq=self.frequency(),
                                   periods=target_limit, closed="left")
            elif side == "after":
                index = date_range(start=self.end(), freq=self.frequency(),
                                   periods=target_limit, closed="right")
            else:
                raise ValueError("side argument isn't valid")

            values = [value] * len(index)
            df = DataFrame(index=index, data=values)
            pad = TimeSeries(df)

        # Create padding TimeSeries from time stamp as str
        if isinstance(limit, str):
            target_limit = Timestamp(limit)
            pad = create_pad(target_limit, self, value)

        # Create padding TimeSeries from time stamp as Pandas Timestamp
        elif isinstance(limit, Timestamp):
            target_limit = limit
            pad = create_pad(target_limit, self, value)

        else:
            ValueError("limit argument isn't valid")

        return self.merge(pad)

    def trim(self, side: str = "both") -> 'TimeSeries':
        """Remove NaNs from a TimeSeries start, end or both

        Args:
            side:
                the side where to remove the NaNs. Valid values are either
                "start", "end" or "both". Default to "both"

        Returns:
            TimeSeries
        """
        if side == "both":
            first = self._data.first_valid_index()
            last = self._data.last_valid_index()
            series_wo_nans = self._data[COMPONENT_VALUES].loc[first:last]
        elif side == "start":
            first = self._data.first_valid_index()
            series_wo_nans = self._data[COMPONENT_VALUES].loc[first:]
        elif side == "end":
            last = self._data.last_valid_index()
            series_wo_nans = self._data[COMPONENT_VALUES].loc[:last]
        else:
            raise AttributeError("side attribute must be either 'start' or "
                                 "'end', but not {}".format(side))
        return TimeSeries(series_wo_nans, self._handler)

    def merge(self, ts: 'TimeSeries') -> 'TimeSeries':
        """Merge two time series and make sure all the given indexes are sorted.

        Args:
            ts: the TimeSeries to merge with self

        Returns:
            TimeSeries
        """
        # append and infer new freq
        merged = self._data.append(ts._data)
        infer_freq(merged.index)

        # instanciate a TimeSeries to sort it
        return TimeSeries(merged, self._handler)

    # ==========================================================================
    # Processing
    # ==========================================================================

    def apply(self, func, ts: 'TimeSeries' = None, keep_handler: bool = True) \
            -> 'TimeSeries':
        """Wrapper around the Pandas apply function

        Args:
            func: function
                Python function or NumPy ufunc to apply. If ts is given as
                param, func must have two params in the form of f(x,y)

            ts: TimeSeries
                The second TimeSeries if you want to make an operation on two
                time series

        Returns:
            TimeSeries
        """
        if ts is not None:

            # The shapes must be the same in order to perform an operation
            # between two time series
            assert self._data.shape == ts._data.shape, \
                "The shape of the TimeSeries given as argument must be " \
                "equal to self."

            df_1 = self._data[self._handler.get_columns()]
            df_2 = ts._data[self._handler.get_columns()]

            vec_func = np.vectorize(func)
            new_data = DataFrame(vec_func(df_1, df_2))

            # TODO Continue here, the index isn't propagated
            print(new_data)

            res = TimeSeries(new_data, self._handler) \
                if keep_handler is True \
                else TimeSeries(new_data, self._handler)

        else:
            res = TimeSeries(self._data[self._handler.get_columns()].apply(func),
                             self._handler)

        return res

    def resample(self, freq: str, method: Optional[str] = None) -> 'TimeSeries':
        """Convert TimeSeries to specified frequency. Optionally provide filling
        method to pad/backfill missing values.

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
            TimeSeries
        """
        # select all element in series that are not duplicates
        new_data = self._data[~self._data.index.duplicated()]
        new_data = new_data.asfreq(freq, method=method)
        return TimeSeries(new_data, self._handler)

    def group_by(self, freq: str, method: Optional[str] = "mean") \
            -> 'TimeSeries':
        """Groups values by a frequency.

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
            TimeSeries
        """
        # Group by freq
        data = self._data.groupby(self.index.round(freq))

        # Apply aggregation
        if method == "mean":
            data = data.mean()
        elif method == "sum":
            data = data.sum()
        elif method == "size":
            data = data.size()
        elif method == "count":
            data = data.count()
        elif method == "std":
            data = data.std()
        elif method == "var":
            data = data.var()
        elif method == "sem":
            data = data.sem()
        elif method == "first":
            data = data.first()
        elif method == "last":
            data = data.last()
        elif method == "min":
            data = data.min()
        elif method == "max":
            data = data.max()
        else:
            ValueError("method argument not recognized.")

        return TimeSeries(data, self._handler)

    def interpolate(self, *args, **kwargs) -> 'TimeSeries':
        """Wrapper around the Pandas interpolate() method.

        See https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.interpolate.html
        for reference

        Args:
            *args: check pandas documentation
            **kwargs: check pandas documentation

        Returns:
            TimeSeries

        """
        new_data = self._data.interpolate(*args, **kwargs)
        return TimeSeries(new_data, self._handler)

    def normalize(self, method: str) -> 'TimeSeries':
        """Normalize a TimeSeries with a given method

        Args:
            method: str
            * 'minmax' for a min max normalization
            * 'zscore' for Z score normalization

        Returns:
            TimeSeries
        """
        if method == "minmax":
            scaled_data = Scaler.minmax(self)
        elif method == "zscore":
            scaled_data = Scaler.zscore(self)
        else:
            raise ValueError("{} isn't recognized as a normalization method"
                             .format(method))
        return scaled_data

    def round(self, decimals: int) -> 'TimeSeries':
        """Round the values in the series.values

        Args:
            decimals: number of digits after the comma

        Returns:
            TimeSeries

        """
        new_data = self._data.astype(float).round(decimals=decimals)
        return TimeSeries(new_data, self._handler)

    def sort(self, *args, **kwargs):
        """Sort a TimeSeries by time stamps

        Basically, it's a wrapper around df.sort_index()
        see: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_index.html

        Args:
            args: the positional arguments
            kwargs: the keyword arguments

        Returns:
            TimeSeries
        """
        new_data = self._data.sort_index(*args, **kwargs)
        return TimeSeries(new_data, self._handler)

    # ==========================================================================
    # Analysis
    # ==========================================================================

    # Basic Statistics
    # ----------------

    def min(self) -> float:
        """Get the minimum value of a TimeSeries

        Returns:
            float
        """
        return self._data[COMPONENT_VALUES].min()

    def max(self) -> float:
        """Get the maximum value of a TimeSeries

        Returns:
            float
        """
        return self._data[COMPONENT_VALUES].max()

    def mean(self) -> float:
        """Get the mean value of a TimeSeries

        Returs:
            float
        """
        return self._data[COMPONENT_VALUES].mean()

    def median(self) -> float:
        """Get the median value of a TimeSeries

        Returns:
            float
        """
        return self._data[COMPONENT_VALUES].median()

    def skewness(self) -> float:
        """Get the skewness of a TimeSeries

        Returns:
            float
        """
        return self._data[COMPONENT_VALUES].skew()

    def kurtosis(self) -> float:
        """Get the kurtosis of a TimeSeries

        Returns:
            float
        """
        return self._data[COMPONENT_VALUES].kurtosis()

    def describe(self, percentiles=None, include=None, exclude=None) -> Series:
        """Describe a TimeSeries with the describe function from Pandas

        Returns:
            Series
        """
        return self._data[COMPONENT_VALUES].describe()

    # Time Series Statistics
    # ----------------------

    def start(self) -> Timestamp:
        """Get the first Timestamp of a TimeSeries

        Returns:
            Timestamp
        """
        start = self._data.index[0]
        return start

    def end(self) -> Timestamp:
        """Get the last Timestamp of a TimeSeries

        Returns:
            Timestamp
        """
        end = self._data.index[-1]
        return end

    def boundaries(self) -> Tuple[Timestamp, Timestamp]:
        """Get a tuple with the TimeSeries first and last index

        Returns:
            a Tuple of Pandas Timestamps
        """
        start = self._data.index[0]
        end = self._data.index[-1]
        return start, end

    def frequency(self) -> Optional[str]:
        """Get the frequency of a TimeSeries

        Returns:
            str or None
            str of the frequency according to the Pandas Offset Aliases
            (https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases)
            None if no discernible frequency
        """
        return self._data.index.inferred_freq

    def time_detlas(self) -> 'TimeSeries':
        """Compute the time difference in seconds between each timestamp
        of a TimeSeries

        Returns:
            TimeSeries
        """
        deltas = self._data.index.to_series().diff().dt.total_seconds()
        ts = TimeSeries(DataFrame(deltas, columns=[COMPONENT_VALUES]),
                        self._handler)
        return ts

    def duration(self) -> Timedelta:
        """Get the duration of the TimeSeries

        Returns:
            Pandas Timedelta
        """
        start, end = self.boundaries()
        return end - start

    # ==========================================================================
    # IO
    # ==========================================================================

    def to_text(self, path: str) -> NoReturn:
        """Export a TimeSeries to text format

        Args:
            path: str of the path where the TimeSeries will be saved

        Returns:
            None
        """
        # Create the time series file
        file_path = "{}/{}.{}".format(path, TIME_SERIES_FILENAME,
                                      TIME_SERIES_EXT)
        ensure_dir(file_path)
        self.__series_to_csv(self._data, file_path)
        if self.metadata is None and self.class_label is not None:
            self.metadata = Metadata(items={'label': self.class_label})
        # Create the metadata file
        if self.metadata is not None:
            if self.class_label is not None:
                self.metadata.add({'label': self.class_label})
            file_path = "{}/{}.{}".format(path, METADATA_FILENAME, METADATA_EXT)
            ensure_dir(file_path)
            self.metadata.to_json(pretty_print=True, path=file_path)

    def to_array(self) -> np.ndarray:
        """Convert a TimeSeries to Numpy Array

        Returns:
            numpy.array with dimensions (1, n), where n is th length of the
            TimeSeries
        """
        return self._data[COMPONENT_VALUES].to_numpy()

    def to_pickle(self, path: str) -> NoReturn:
        """Export a TimeSeries to Pickle

        Args:
            path: str of the path where the TimeSeries will be saved
        """
        to_pickle(self, path)

    def to_darts(self) -> DartsTimeSeries:
        """Convert a TimeSeries to Darts TimeSeries

        Returns:
            Darts TimeSeries object
        """
        return DartsTimeSeries.from_series(self._data[COMPONENT_VALUES])

    def to_df(self) -> DataFrame:
        """Converts a TimeSeries to a Pandas DataFrame

        Returns:
            Pandas DataFrame
        """
        return self._data

    @staticmethod
    def __series_to_csv(series: Union[Series, DataFrame], path: str) -> NoReturn:
        """Export a Pandas Series or DataFrame and put it into a CSV file

        Args:
            series: Pandas Series or Pandas DataFrame
                The data to write in CSV
            path: str
                The path where the Series will be saved
        """
        series.to_csv(path, header=True, index_label="index")

    @staticmethod
    def __list_diff(list_1: List, list_2: List):
        """ Compute the difference between two lists

        Args:
            list_1: first list
            list_2: second list

        Returns:
            List
        """
        return [i for i in list_1 + list_2
                if i not in list_1 or i not in list_2]
