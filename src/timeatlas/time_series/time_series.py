from typing import NoReturn, Tuple, Any, Union, Optional, List, Callable

from darts import TimeSeries as DartsTimeSeries
import numpy as np
from pandas import DataFrame, date_range, infer_freq, Series, DatetimeIndex, \
    Timestamp, Timedelta

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
from timeatlas.plots.time_series import line_plot, status_plot
from timeatlas.utils import ensure_dir, to_pickle


class TimeSeries(AbstractBaseTimeSeries, AbstractOutputText,
                 AbstractOutputPickle):
    """Defines a time series

    A TimeSeries object is a series of time indexed values.

    Attributes:
        series: An optional Pandas DataFrame
        metadata: An optional Dict storing metadata about this TimeSeries
    """

    def __init__(self, series: Union[Series, DataFrame] = None,
                 metadata: Metadata = None, label: str or None = None):

        if series is not None:
            # Check if values have a DatetimeIndex
            assert isinstance(series.index, DatetimeIndex), \
                'Values must be indexed with a DatetimeIndex.'

            # Check if the length is greater than one
            assert len(series) >= 1, 'Values must have at least one values.'

            # Save the data with the right format
            if isinstance(series, Series):
                series.name = TIME_SERIES_VALUES
                series = series.to_frame()

            elif isinstance(series, DataFrame):
                assert len(series.columns) >= 1, \
                    "DataFrame as input series must have at least one column."

                # If there's only one column, then it is the values column
                if len(series.columns) == 1:
                    series.columns = [TIME_SERIES_VALUES]

                # Otherwise, one column should be called "values"
                assert TIME_SERIES_VALUES in series.columns, \
                    "DataFrame as input series must contain a column called {}"\
                    .format(TIME_SERIES_VALUES)

            # Create the TimeSeries object with certainty that values
            # are sorted by the time index
            self.series = series.sort_index()

            # Add the freq if regular
            if len(series) >= 3:
                infer_freq(self.series.index)

            # Create instance variables
            self.index = self.series.index  # index accessor
            self.values = self.series[TIME_SERIES_VALUES]  # values accessor
        else:
            self.series = None

        self.label = label  # label of the TimeSeries (for classification)

        if metadata is not None:
            self.metadata = metadata
        else:
            self.metadata = None

    def __repr__(self):
        return self.series.__repr__()

    def __len__(self):
        return len(self.series)

    def __iter__(self):
        return (v for v in self.series[TIME_SERIES_VALUES])

    def __getitem__(self, item):
        return TimeSeries(self.series[item])

    def __setitem__(self, item, value):
        if isinstance(value, TimeSeries):
            self.series[item] = value.series[item]
        else:
            self.series[item] = value[item]

    # ==========================================================================
    # Methods
    # ==========================================================================

    @staticmethod
    def create(start: str, end: str, freq: Union[str, 'TimeSeries'] = None,
               metadata: Metadata = None) \
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
                freq = infer_freq(freq.series.index)
            elif isinstance(freq, str):
                freq = freq
        series = DataFrame(columns=[TIME_SERIES_VALUES],
                           index=date_range(start, end, freq=freq))
        return TimeSeries(series, metadata)

    def plot(self) -> Any:
        """Plot a TimeSeriesDataset

        Returns:
            plotly.graph_objects.Figure
        """
        return line_plot(self)

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
        start = self.series.index[0]
        end = self.series.index[-1]
        first_split = self.series[start:timestamp].copy()
        second_split = self.series[timestamp:end].copy()
        before = TimeSeries(first_split, self.metadata)
        after = TimeSeries(second_split, self.metadata)
        return before, after

    def split_in_chunks(self, n: int) -> List['TimeSeries']:
        """Split a TimeSeries into chunks of length n

        Args:
            n: length of the chunks

        Returns:
            List of TimeSeries
        """
        ts_chunks = [TimeSeries(series=v, metadata=self.metadata) for n, v in
                     self.series.groupby(np.arange(len(self.series)) // n)]
        return ts_chunks

    def fill(self, value: Any) -> 'TimeSeries':
        """Fill a TimeSeries with a value. If given a unique value, all values
        will be broadcasted. If given an array of the length of the TimeSeries,
        it will replace all values.

        Args:
            value: Any values that you want to fill the TimeSeries with

        Returns:
            TimeSeries
        """
        s = self.series.copy()
        s[TIME_SERIES_VALUES] = value
        return TimeSeries(s, self.metadata)

    def empty(self) -> 'TimeSeries':
        """Empty the TimeSeries (fill all values with NaNs)

        Returns:
            TimeSeries
        """
        return self.fill(np.nan)

    def pad(self, limit: Union[int, str, Timestamp], side: Optional[str] = None,
            value: Any = np.NaN):
        """
        Pad a TimeSeries until a given limit

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
                return ts.create(new_limit, ts.start(), freq=ts)\
                           .fill(fill_val)[:-1]
            elif new_limit > ts.end():
                return ts.create(ts.end(), new_limit, freq=ts)\
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
            first = self.series.first_valid_index()
            last = self.series.last_valid_index()
            series_wo_nans = self.series[TIME_SERIES_VALUES].loc[first:last]
        elif side == "start":
            first = self.series.first_valid_index()
            series_wo_nans = self.series[TIME_SERIES_VALUES].loc[first:]
        elif side == "end":
            last = self.series.last_valid_index()
            series_wo_nans = self.series[TIME_SERIES_VALUES].loc[:last]
        else:
            raise AttributeError("side attribute must be either 'start' or "
                                 "'end', but not {}".format(side))
        return TimeSeries(series_wo_nans, self.metadata)

    def merge(self, ts: 'TimeSeries') -> 'TimeSeries':
        """Merge two time series and make sure all the given indexes are sorted.

        Args:
            ts: the TimeSeries to merge with self

        Returns:
            TimeSeries
        """
        # append and infer new freq
        merged = self.series.append(ts.series)
        infer_freq(merged.index)

        # instanciate a TimeSeries to sort it
        return TimeSeries(merged, self.metadata)

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
            s1 = self.series[TIME_SERIES_VALUES]
            s2 = ts.series[TIME_SERIES_VALUES]
            df = DataFrame(data={"s1": s1, "s2": s2})
            res = TimeSeries(df.apply(lambda x: func(x.s1, x.s2), axis=1),
                             self.metadata)
        else:
            res = TimeSeries(self.series[TIME_SERIES_VALUES].apply(func),
                             self.metadata)

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
        # avoid duplicated indexes
        new_series = self.series[~self.series.index.duplicated()]
        new_series = new_series.asfreq(freq, method=method)
        return TimeSeries(new_series, self.metadata)

    def group_by(self, freq: str, method: Optional[str] = "mean")\
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
        series = self.series.groupby(self.index.round(freq))

        # Apply aggregation
        if method == "mean":
            series = series.mean()
        elif method == "sum":
            series = series.sum()
        elif method == "size":
            series = series.size()
        elif method == "count":
            series = series.count()
        elif method == "std":
            series = series.std()
        elif method == "var":
            series = series.var()
        elif method == "sem":
            series = series.sem()
        elif method == "first":
            series = series.first()
        elif method == "last":
            series = series.last()
        elif method == "min":
            series = series.min()
        elif method == "max":
            series = series.max()
        else:
            ValueError("method argument not recognized.")

        return TimeSeries(series, self.metadata)

    def interpolate(self, *args, **kwargs) -> 'TimeSeries':
        """Wrapper around the Pandas interpolate() method.

        See https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.interpolate.html
        for reference
        """
        new_series = self.series.interpolate(*args, **kwargs)
        return TimeSeries(new_series, self.metadata)

    def normalize(self, method: str) -> 'TimeSeries':
        """Normalize a Time Series with a given method

        Args:
            method: str
            * 'minmax' for a min max normalization
            * 'zscore' for Z score normalization

        Returns:
            TimeSeries
        """
        if method == "minmax":
            scaled_series = Scaler.minmax(self.series)
        elif method == "zscore":
            scaled_series = Scaler.zscore(self.series)
        else:
            raise ValueError("{} isn't recognized as a normalization method"
                             .format(method))
        return TimeSeries(scaled_series, self.metadata)

    def round(self, decimals: int) -> 'TimeSeries':
        """Round the values in the series.values

        Args:
            decimals: number of digits after the comma
        """
        return TimeSeries(self.series.astype(float).round(decimals=decimals),
                          metadata=self.metadata)

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
        sorted_time_series = self.series.sort_index(*args, **kwargs)
        return TimeSeries(sorted_time_series, self.metadata)

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
        return self.series[TIME_SERIES_VALUES].min()

    def max(self) -> float:
        """Get the maximum value of a TimeSeries

        Returns:
            float
        """
        return self.series[TIME_SERIES_VALUES].max()

    def mean(self) -> float:
        """Get the mean value of a TimeSeries

        Returs:
            float
        """
        return self.series[TIME_SERIES_VALUES].mean()

    def median(self) -> float:
        """Get the median value of a TimeSeries

        Returns:
            float
        """
        return self.series[TIME_SERIES_VALUES].median()

    def skewness(self) -> float:
        """Get the skewness of a TimeSeries

        Returns:
            float
        """
        return self.series[TIME_SERIES_VALUES].skew()

    def kurtosis(self) -> float:
        """Get the kurtosis of a TimeSeries

        Returns:
            float
        """
        return self.series[TIME_SERIES_VALUES].kurtosis()

    def describe(self, percentiles=None, include=None, exclude=None) -> Series:
        """Describe a TimeSeries with the describe function from Pandas

        Returns:
            Series
        """
        return self.series[TIME_SERIES_VALUES].describe()

    # Time Series Statistics
    # ----------------------

    def start(self) -> Timestamp:
        """Get the first Timestamp of a TimeSeries

        Returns:
            Timestamp
        """
        start = self.series.index[0]
        return start

    def end(self) -> Timestamp:
        """Get the last Timestamp of a TimeSeries

        Returns:
            Timestamp
        """
        end = self.series.index[-1]
        return end

    def boundaries(self) -> Tuple[Timestamp, Timestamp]:
        """Get a tuple with the TimeSeries first and last index

        Returns:
            a Tuple of Pandas Timestamps
        """
        start = self.series.index[0]
        end = self.series.index[-1]
        return start, end

    def frequency(self) -> Optional[str]:
        """Get the frequency of a TimeSeries

        Returns:
            str or None
            str of the frequency according to the Pandas Offset Aliases
            (https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases)
            None if no discernible frequency
        """
        return self.series.index.inferred_freq

    def resolution(self) -> 'TimeSeries':
        """Compute the time difference between each timestamp of a TimeSeries

        Returns:
            TimeSeries
        """
        diff = self.series.index.to_series().diff().dt.total_seconds()
        ts = TimeSeries(DataFrame(diff, columns=[TIME_SERIES_VALUES]),
                        self.metadata)
        return ts

    def duration(self) -> Timedelta:
        """Get the duration of the TimeSeries

        Returns:
            a Pandas Timedelta
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
        """
        # Create the time series file
        file_path = "{}/{}.{}".format(path, TIME_SERIES_FILENAME,
                                      TIME_SERIES_EXT)
        ensure_dir(file_path)
        self.__series_to_csv(self.series, file_path)
        if self.metadata is None and self.label is not None:
            self.metadata = Metadata(items={'label': self.label})
        # Create the metadata file
        if self.metadata is not None:
            if self.label is not None:
                self.metadata.add({'label': self.label})
            file_path = "{}/{}.{}".format(path, METADATA_FILENAME, METADATA_EXT)
            ensure_dir(file_path)
            self.metadata.to_json(pretty_print=True, path=file_path)

    def to_array(self) -> np.ndarray:
        """Convert a TimeSeries to Numpy Array

        Returns:
            numpy.array with dimensions (1, n), where n is th length of the
            TimeSeries
        """
        return self.series[TIME_SERIES_VALUES].to_numpy()

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
        return DartsTimeSeries.from_series(self.series[TIME_SERIES_VALUES])

    def to_df(self) -> DataFrame:
        """Converts a TimeSeries to a Pandas DataFrame

        Returns:
            Pandas DataFrame
        """
        return self.series

    @staticmethod
    def __series_to_csv(series: Union[Series, DataFrame], path: str) \
            -> NoReturn:
        """Export a Pandas Series or DataFrame and put it into a CSV file

        Args:
            series: Pandas Series or Pandas DataFrame
                The data to write in CSV
            path: str
                The path where the Series will be saved
        """
        series.to_csv(path, header=True, index_label="index")
