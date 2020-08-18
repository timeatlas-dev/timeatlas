from pandas import DataFrame, date_range, infer_freq, Series, DatetimeIndex, \
    Timestamp, Timedelta, concat
from pandas.plotting import register_matplotlib_converters
from typing import NoReturn, Tuple, Any, Union, Optional

from darts import TimeSeries as DartsTimeSeries

from timeatlas.abstract.abstract_analysis import AbstractAnalysis
from timeatlas.abstract import AbstractOutputText, AbstractOutputPickle
from timeatlas.abstract.abstract_processing import AbstractProcessing
from timeatlas.config.constants import (
    TIME_SERIES_VALUES,
    TIME_SERIES_FILENAME,
    TIME_SERIES_EXT,
    METADATA_FILENAME,
    METADATA_EXT
)
from timeatlas.metadata import Metadata
from timeatlas.processors.scaler import Scaler
from timeatlas.utils import ensure_dir, to_pickle


class TimeSeries(AbstractAnalysis, AbstractOutputText,
                 AbstractOutputPickle, AbstractProcessing):
    """ Defines a time series

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
                    "DataFrame as input series must contain a column called {}"\
                    .format(TIME_SERIES_VALUES)

        # Create the TimeSeries object
        self.series = series

        # The label of the timeseries (can be used for the classification)
        self.label = label

        if metadata is not None:
            self.metadata = metadata
        else:
            self.metadata = None

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
               metadata: Metadata = None):
        """
        Creates an empty TimeSeries object with the period as index

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

    def split(self, splitting_point: str) -> Tuple['TimeSeries', 'TimeSeries']:
        """
        Split a TimeSeries at a defined point and include the splitting point
        in both as in [start,...,at] and [at,...,end].

        Args:
            splitting_point: str where to the TimeSeries will be split
                            (e.g. "2019-12-31 00:00:00")

        Returns:
            a Tuple of TimeSeries ([start,...,at] and [at,...,end])

        """
        start = self.series.index[0]
        end = self.series.index[-1]
        first_split = self.series[start:splitting_point].copy()
        second_split = self.series[splitting_point:end].copy()
        before = TimeSeries(first_split, self.metadata)
        after = TimeSeries(second_split, self.metadata)
        return before, after

    def fill(self, value: Any) -> 'TimeSeries':
        """
        Fill a TimeSeries with a value. If given a unique value, all values will
        be broadcasted. If given an array of the length of the TimeSeries, it
        will replace all values.

        :param value: Any values that you want to fill the TimeSeries with
        :return: TimeSeries
        """
        s = self.series.copy()
        s[TIME_SERIES_VALUES] = value
        return TimeSeries(s, self.metadata)

    def empty(self) -> 'TimeSeries':
        """
        Empty the TimeSeries (fill all values with NaNs)

        Returns:
            TimeSeries
        """
        return self.fill(None)

    def trim(self, side: str = "both"):
        """
        Remove NaNs from a TimeSeries start, end or both

        Args:
            side: the side where to remove the NaNs. Valid values are either
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

    def append(self, ts: 'TimeSeries'):
        """
        Append a TimeSeries to another TimeSeries

        :param ts: TimeSeries to append
        :return: TimeSeries
        """
        new_series = self.series.append(ts.series)
        return TimeSeries(new_series, self.metadata)

    def prepend(self, ts: 'TimeSeries'):
        """
        Prepend a TimeSeries to another TimeSeries

        :param ts: TimeSeries to prepend
        :return: TimeSeries
        """
        new_series = concat([ts.series, self.series])
        return TimeSeries(new_series, self.metadata)

    # ==========================================================================
    # Analysis
    # ==========================================================================

    def plot(self, *args, **kwargs):
        """
        Plot a TimeSeries

        This is a wrapper around Pandas.Series.plot() augmented if the
        TimeSeries to plot has associated Metadata.

        :param args: positional arguments for Pandas plot() method
        :param kwargs: keyword arguments fot Pandas plot() method
        """
        register_matplotlib_converters()

        if 'figsize' not in kwargs:
            kwargs['figsize'] = (18, 2)  # Default TimeSeries plot format

        if 'color' not in kwargs:
            kwargs['color'] = "k"

        ax = self.series.plot(*args, **kwargs)
        ax.set_xlabel("Date")
        ax.grid(True, c='gray', ls='-', lw=1, alpha=0.2)

        # Add legend from metadata if existing
        if self.metadata is not None:
            if "unit" in self.metadata:
                unit = self.metadata["unit"]
                ax.set_ylabel("{} $[{}]$".format(unit.name, unit.symbol))
            if "sensor" in self.metadata:
                sensor = self.metadata["sensor"]
                ax.set_title("{}—{}".format(sensor.id, sensor.name))

    def describe(self, percentiles=None, include=None, exclude=None) -> Series:
        """
        Describe a TimeSeries with the describe function from Pandas

        Returns:
            Series
        """
        return self.series.describe()

    def min(self) -> float:
        """
        Get the minimum value of a TimeSeries

        Returns:
            float
        """
        return self.series.min()

    def max(self) -> float:
        """
        Get the maximum value of a TimeSeries

        Returns:
            float
        """
        return self.series.max()

    def mean(self) -> float:
        """
        Get the mean value of a TimeSeries

        Returs:
            float
        """
        return self.series.mean()

    def median(self) -> float:
        """
        Get the median value of a TimeSeries

        Returns:
            float
        """
        return self.series.median()

    def skewness(self) -> float:
        """
        Get the skewness of a TimeSeries

        Returns:
            float
        """
        return self.series.skew()

    def kurtosis(self) -> float:
        """
        Get the kurtosis of a TimeSeries

        Returns:
            float
        """
        return self.series.kurtosis()

    def boundaries(self) -> Tuple[Timestamp, Timestamp]:
        """
        Get a tuple with the TimeSeries first and last index

        Returns:
            a Tuple of Pandas Timestamps
        """
        start = self.series.index[0]
        end = self.series.index[-1]
        return start, end

    def duration(self) -> Timedelta:
        """
        Get the duration of the TimeSeries

        Returns:
            a Pandas Timedelta
        """
        start, end = self.boundaries()
        return end - start

    def frequency(self) -> Optional[str]:
        """
        Get the frequency of a TimeSeries

        Returns:
            str or None
            str of the frequency according to the Pandas Offset Aliases
            (https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
            None if no discernible frequency
        """
        return self.series.index.inferred_freq

    def resolution(self) -> 'TimeSeries':
        """
        Compute the time difference between each timestamp of a TimeSeries

        Returns:
            TimeSeries
        """
        diff = self.series.index.to_series().diff()
        return TimeSeries(DataFrame(diff, columns=[TIME_SERIES_VALUES]),
                          self.metadata)

    # ==========================================================================
    # Processing
    # ==========================================================================

    def sort(self, *args, **kwargs):
        """
        Sort a TimeSeries by time stamps

        Basically, it's a wrapper around df.sort_index()
        see: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_index.html

        :param args:
        :param kwargs:
        :return: TimeSeries
        """
        sorted_time_series = self.series.sort_index(*args, **kwargs)
        return TimeSeries(sorted_time_series, self.metadata)

    def apply(self, func, ts: 'TimeSeries' = None):
        """
        Wrapper around the Pandas apply function

        Args:
            func : function
                Python function or NumPy ufunc to apply. If ts is given as
                param, func must have two params in the form of f(x,y)

            ts : TimeSeries
                The second TimeSeries if you want to make an operation on two
                time series

            convert_dtype : bool, default True
                Try to find better dtype for elementwise function results. If
                False, leave as dtype=object.

            args : tuple
                Positional arguments passed to func after the series value.

            **kwds
                Additional keyword arguments passed to func.

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
            res = TimeSeries(self.series.apply(func),
                             self.metadata)
        return res

    def resample(self, freq: str, method: Optional[str] = None) -> 'TimeSeries':
        """
        Convert TimeSeries to specified frequency. Optionally provide filling
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
        """
        Wrapper around the Pandas interpolate() method.

        See https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.interpolate.html
        for reference

        """
        new_series = self.series.interpolate(*args, **kwargs)
        return TimeSeries(new_series, self.metadata)

    def normalize(self, method: str) -> 'TimeSeries':
        """
        Normalize a Time Series with a given method

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
        """
        Round the values in the series.values

        Args:
            decimals: number of digits after the comma
        """
        return TimeSeries(self.series.astype(float).round(decimals=decimals),
                          metadata=self.metadata)

    # ==========================================================================
    # IO
    # ==========================================================================

    def to_text(self, path: str) -> NoReturn:
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

    def to_pickle(self, path: str) -> NoReturn:
        to_pickle(self, path)

    def to_darts(self) -> DartsTimeSeries:
        """ TimeAtlas TimeSeries to Darts TimeSeries
        conversion method

        Returns: Darts TimeSeries object
        """
        return DartsTimeSeries.from_series(self.series[TIME_SERIES_VALUES])

    def to_df(self) -> DataFrame:
        """ TimeAtlas TimeSeries to Pandas DataFrame
        conversion method

        Returns: Pandas DataFrame
        """
        return self.series

    @staticmethod
    def __series_to_csv(series: Union[Series, DataFrame], path: str):
        """
        Read a Pandas Series and put it into a CSV file

        Args:
            series: The Series to write in CSV
            path: The path where the Series will be saved
        """
        series.to_csv(path, header=True, index_label="index")
