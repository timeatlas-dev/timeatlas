from typing import NoReturn, Tuple, Any, Union, Optional, List
from copy import deepcopy, copy

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
    TIME_SERIES_VALUES,
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

    def __init__(self, data: DataFrame = None,
                 components: ComponentHandler = None):
        """Defines a time series

        A TimeSeries object is a series of time indexed values.

        Args:
            data: DataFrame containing the values and labels
            components: ComponentHandler
        """
        if data is not None:

            # Perform preliminary checks
            # --------------------------

            assert isinstance(data.index, DatetimeIndex), \
                'Values must be indexed with a DatetimeIndex.'

            assert len(data) >= 1, \
                'Values must have at least one values.'

            assert len(data.columns) >= 1, \
                "DataFrame must have at least one column."

            # Create the TimeSeries object
            # ----------------------------

            # Create the components handler
            if components is None:
                self.components = ComponentHandler()
                for col in data.columns:
                    component = Component(col)
                    self.components.append(component)
            else:
                self.components = components

            # Rename the columns
            data.columns = self.components.get_components()

            # Store the data with certainty that values are sorted
            self.data = data.sort_index()

            # Add the freq if regular
            if len(data) >= 3:
                self.data.index.freq = infer_freq(self.data.index)

            # Create instance variables
            self.index = self.data.index  # index accessor
            self.values = self.data[self.components.get_values()]  # values accessor

        else:

            # Create empty structures
            self.data = DataFrame()
            self.components = ComponentHandler()

    def __repr__(self):
        return self.data.__repr__()

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return (v for v in self.data[TIME_SERIES_VALUES])

    def __getitem__(self, item: Union[Tuple, Any]):
        if isinstance(item, Tuple):
            time = item[0]
            component = item[1]
            new_data = self.data.loc[time].iloc[:, component]
            new_components = self.components[component]
            return TimeSeries(new_data, new_components)
        else:
            new_data = self.data[item]
            return TimeSeries(new_data, self.components)

    def __setitem__(self, item, value):
        if isinstance(value, TimeSeries):
            self.data[item] = value.data[item]
        else:
            self.data[item] = value[item]

    # ==========================================================================
    # Methods
    # ==========================================================================

    @staticmethod
    def create(start: str, end: str, freq: Union[str, 'TimeSeries'] = None,
               components: ComponentHandler = None) \
            -> 'TimeSeries':
        """Creates an empty TimeSeries object with the period as index

        Args:
            start: str of the start of the DatetimeIndex (as in Pandas.date_range())
            end: the end of the DatetimeIndex (as in Pandas.date_range())
            freq: the optional frequency it can be a str or a TimeSeries (to copy its frequency)
            metadata: the optional Metadata object

        Returns:
            TimeSeries
        """
        if freq is not None:
            if isinstance(freq, TimeSeries):
                freq = infer_freq(freq.data.index)
            elif isinstance(freq, str):
                freq = freq
        data = DataFrame(columns=[TIME_SERIES_VALUES],
                         index=date_range(start, end, freq=freq))
        return TimeSeries(data, components)

    def add_component(self, ts: 'TimeSeries'):
        assert (self.index == ts.index).all(), \
            "Indexes aren't the same"
        new_data = concat([self.data, ts.data], axis=1)
        new_components = self.components.copy()
        for c in ts.components.components:
            new_components.append(c)
        return TimeSeries(new_data, new_components)

    def add_label(self, ts: 'TimeSeries'):
        pass

    def add_ci(self, ts: 'TimeSeries'):
        pass

    def add_meta(self, ts: 'TimeSeries'):
        pass

    def remove_component(self, index: int):
        new_data = self.data.copy()
        new_data = new_data.drop(new_data.columns[index], axis=1)
        new_components = self.components.copy()
        del new_components[index]
        return TimeSeries(new_data, new_components)

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
        start = self.data.index[0]
        end = self.data.index[-1]
        first_split = self.data[start:timestamp].copy()
        second_split = self.data[timestamp:end].copy()
        before = TimeSeries(first_split, self.components)
        after = TimeSeries(second_split, self.components)
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
        ts_chunks = [TimeSeries(data=v, components=self.components) for n, v in
                     self.data.groupby(np.arange(len(self.data)) // n)]
        return ts_chunks

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
        new_data = self.data.copy()
        new_data[:] = value
        return TimeSeries(new_data, self.components)

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
            first = self.data.first_valid_index()
            last = self.data.last_valid_index()
            series_wo_nans = self.data[TIME_SERIES_VALUES].loc[first:last]
        elif side == "start":
            first = self.data.first_valid_index()
            series_wo_nans = self.data[TIME_SERIES_VALUES].loc[first:]
        elif side == "end":
            last = self.data.last_valid_index()
            series_wo_nans = self.data[TIME_SERIES_VALUES].loc[:last]
        else:
            raise AttributeError("side attribute must be either 'start' or "
                                 "'end', but not {}".format(side))
        return TimeSeries(series_wo_nans, self.components)

    def merge(self, ts: 'TimeSeries') -> 'TimeSeries':
        """Merge two time series and make sure all the given indexes are sorted.

        Args:
            ts: the TimeSeries to merge with self

        Returns:
            TimeSeries
        """
        # append and infer new freq
        merged = self.data.append(ts.data)
        infer_freq(merged.index)

        # instanciate a TimeSeries to sort it
        return TimeSeries(merged, self.components)

    # ==========================================================================
    # Processing
    # ==========================================================================

    def apply(self, func, ts: 'TimeSeries' = None) -> 'TimeSeries':
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

            assert len(self) == len(ts), "The length of the TimeSeries given " \
                                         "as argument should be equal to self."

            s1 = self.data[TIME_SERIES_VALUES]
            s2 = ts.data[TIME_SERIES_VALUES]
            df = DataFrame(data={"s1": s1, "s2": s2})
            res = TimeSeries(df.apply(lambda x: func(x.s1, x.s2), axis=1),
                             self.components)
        else:
            res = TimeSeries(self.data[TIME_SERIES_VALUES].apply(func),
                             self.components)

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
        new_data = self.data[~self.data.index.duplicated()]
        new_data = new_data.asfreq(freq, method=method)
        return TimeSeries(new_data, self.components)

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
        data = self.data.groupby(self.index.round(freq))

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

        return TimeSeries(data, self.components)

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
        new_data = self.data.interpolate(*args, **kwargs)
        return TimeSeries(new_data, self.components)

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
        new_data = self.data.astype(float).round(decimals=decimals)
        return TimeSeries(new_data, self.components)

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
        new_data = self.data.sort_index(*args, **kwargs)
        return TimeSeries(new_data, self.components)

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
        return self.data[TIME_SERIES_VALUES].min()

    def max(self) -> float:
        """Get the maximum value of a TimeSeries

        Returns:
            float
        """
        return self.data[TIME_SERIES_VALUES].max()

    def mean(self) -> float:
        """Get the mean value of a TimeSeries

        Returs:
            float
        """
        return self.data[TIME_SERIES_VALUES].mean()

    def median(self) -> float:
        """Get the median value of a TimeSeries

        Returns:
            float
        """
        return self.data[TIME_SERIES_VALUES].median()

    def skewness(self) -> float:
        """Get the skewness of a TimeSeries

        Returns:
            float
        """
        return self.data[TIME_SERIES_VALUES].skew()

    def kurtosis(self) -> float:
        """Get the kurtosis of a TimeSeries

        Returns:
            float
        """
        return self.data[TIME_SERIES_VALUES].kurtosis()

    def describe(self, percentiles=None, include=None, exclude=None) -> Series:
        """Describe a TimeSeries with the describe function from Pandas

        Returns:
            Series
        """
        return self.data[TIME_SERIES_VALUES].describe()

    # Time Series Statistics
    # ----------------------

    def start(self) -> Timestamp:
        """Get the first Timestamp of a TimeSeries

        Returns:
            Timestamp
        """
        start = self.data.index[0]
        return start

    def end(self) -> Timestamp:
        """Get the last Timestamp of a TimeSeries

        Returns:
            Timestamp
        """
        end = self.data.index[-1]
        return end

    def boundaries(self) -> Tuple[Timestamp, Timestamp]:
        """Get a tuple with the TimeSeries first and last index

        Returns:
            a Tuple of Pandas Timestamps
        """
        start = self.data.index[0]
        end = self.data.index[-1]
        return start, end

    def frequency(self) -> Optional[str]:
        """Get the frequency of a TimeSeries

        Returns:
            str or None
            str of the frequency according to the Pandas Offset Aliases
            (https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases)
            None if no discernible frequency
        """
        return self.data.index.inferred_freq

    def time_detlas(self) -> 'TimeSeries':
        """Compute the time difference in seconds between each timestamp
        of a TimeSeries

        Returns:
            TimeSeries
        """
        deltas = self.data.index.to_series().diff().dt.total_seconds()
        ts = TimeSeries(DataFrame(deltas, columns=[TIME_SERIES_VALUES]),
                        self.components)
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
        self.__series_to_csv(self.data, file_path)
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
        return self.data[TIME_SERIES_VALUES].to_numpy()

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
        return DartsTimeSeries.from_series(self.data[TIME_SERIES_VALUES])

    def to_df(self) -> DataFrame:
        """Converts a TimeSeries to a Pandas DataFrame

        Returns:
            Pandas DataFrame
        """
        return self.data

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
