from pandas import DataFrame, date_range, infer_freq, Series, DatetimeIndex, \
    Timestamp, Timedelta, concat
from pandas.plotting import register_matplotlib_converters
from typing import NoReturn, Tuple, Any, Union, Optional, List, Callable
import numpy as np

from darts import TimeSeries as DartsTimeSeries

from timeatlas.abstract.abstract_base_time_series import AbstractBaseTimeSeries
from timeatlas.abstract import AbstractOutputText, AbstractOutputPickle
from timeatlas.config.constants import (
    TIME_SERIES_VALUES,
    TIME_SERIES_FILENAME,
    TIME_SERIES_EXT,
    METADATA_FILENAME,
    METADATA_EXT
)
from timeatlas.metadata import Metadata
from timeatlas.processors.scaler import Scaler
from timeatlas.plots.time_series import line, status
from timeatlas.utils import ensure_dir, to_pickle

from numpy import ndarray


class TimeSeries(AbstractBaseTimeSeries, AbstractOutputText,
                 AbstractOutputPickle):
    """Defines a time series

    A TimeSeries object is a series of time indexed values.

    Attributes:
        series: An optional Pandas DataFrame
        metadata: An optional Dict storing metadata about this TimeSeries
    """

    def __init__(self, series: Union[Series, DataFrame] = None,
                 metadata: Metadata = None, label: str or None = None,
    ):

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
                    "DataFrame as input series must contain a column called {}" \
                        .format(TIME_SERIES_VALUES)

        # Create the TimeSeries object
        self.series = series

        # The label of the timeseries (can be used for the classification)
        self.label = label

        if metadata is not None:
            self.metadata = metadata
        else:
            self.metadata = None

        # Define default plotting function
        self.plotting_function = line

    def __repr__(self):
        return self.series.__repr__()

    def __len__(self):
        return len(self.series)

    def __iter__(self):
        return (v for v in self.series)

    def __getitem__(self, item):
        return TimeSeries(self.series[item])

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

    def register_plotting_function(self, plotting_function: Callable) \
            -> NoReturn:
        """Register a specific plotting function for this TimeSeries

        Args:
            plotting_function: Callable (like a function) that takes at least
                a TimeSeries as first argument
        """
        self.plotting_function = plotting_function

    def plot(self, *args, **kwargs) -> Any:
        """Plot the TimeSeries with the registered plotting function
        (in self.plotting_function)

        Args:
            *args: Arguments to give to the plotting function
            **kwargs: Keyword arguments to give to the plotting function
        """
        assert self.plotting_function is not None, \
            "No plotting function registered"
        self.plotting_function(self, *args, **kwargs)

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
        return self.fill(None)

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
        # TODO
        raise NotImplementedError

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
        new_series = self.series.asfreq(freq, method=method)
        return TimeSeries(new_series, self.metadata)

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

        :param args:
        :param kwargs:
        :return: TimeSeries
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
        return self.series.min()

    def max(self) -> float:
        """Get the maximum value of a TimeSeries

        Returns:
            float
        """
        return self.series.max()

    def mean(self) -> float:
        """Get the mean value of a TimeSeries

        Returs:
            float
        """
        return self.series.mean()

    def median(self) -> float:
        """Get the median value of a TimeSeries

        Returns:
            float
        """
        return self.series.median()

    def skewness(self) -> float:
        """Get the skewness of a TimeSeries

        Returns:
            float
        """
        return self.series.skew()

    def kurtosis(self) -> float:
        """Get the kurtosis of a TimeSeries

        Returns:
            float
        """
        return self.series.kurtosis()

    def describe(self, percentiles=None, include=None, exclude=None) -> Series:
        """Describe a TimeSeries with the describe function from Pandas

        Returns:
            Series
        """
        return self.series.describe()

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
        raise end

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
        ts.register_plotting_function(lambda x: status(x, cmap="prism"))
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
        # Create the metadata file
        if self.metadata is not None:
            file_path = "{}/{}.{}".format(path, METADATA_FILENAME, METADATA_EXT)
            ensure_dir(file_path)
            self.metadata.to_json(pretty_print=True, path=file_path)

    def to_array(self) -> ndarray:
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
