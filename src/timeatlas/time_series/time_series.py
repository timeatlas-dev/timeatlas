from pandas import DataFrame, date_range, infer_freq, Series, DatetimeIndex, Timestamp, Timedelta
from pandas.plotting import register_matplotlib_converters
from typing import NoReturn, List, Tuple, Any, Union, Optional

from u8timeseries import TimeSeries as U8TimeSeries

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

    # Methods
    # =======

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
            splitting_point: str where to the TimeSeries will be split (e.g. "2019-12-31 00:00:00")

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

    def erase(self) -> 'TimeSeries':
        """
        Empty the TimeSeries (fill all values with NaNs)

        Returns:
            TimeSeries
        """
        s = self.series.copy()
        s.values[:] = None
        return TimeSeries(s, self.metadata)

    # =============================================
    # Analysis
    # =============================================
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
            kwargs['figsize'] = (18,2) # Default TimeSeries plot format

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

    # =============================================
    # Processing
    # =============================================

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

    def resample(self, by: str) -> Any:
        """
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
        - upsampling
        - downsampling
        """
        pass

    def interpolate(self, method: str) -> Any:
        """
        Intelligent interpolation in function of the data unit etc.
        """
        pass

    def normalize(self, method: str) -> Any:
        """
        Normalize a dataset
        """
        pass

    def round(self, decimals: int) -> 'TimeSeries':
        """
        Round the values in the series.values

        Args:
            decimals: number of digits after the comma
        """
        return TimeSeries(self.series.astype(float).round(decimals=decimals),
                          metadata=self.metadata)

    # =============================================
    # IO
    # =============================================

    def to_text(self, path: str) -> NoReturn:
        # Create the time series file
        file_path = "{}/{}.{}".format(path, TIME_SERIES_FILENAME, TIME_SERIES_EXT)
        ensure_dir(file_path)
        self.__series_to_csv(self.series, file_path)
        # Create the metadata file
        if self.metadata is not None:
            file_path = "{}/{}.{}".format(path, METADATA_FILENAME, METADATA_EXT)
            ensure_dir(file_path)
            self.metadata.to_json(pretty_print=True, path=file_path)

    def to_pickle(self, path: str) -> NoReturn:
        to_pickle(self, path)

    def to_u8(self) -> U8TimeSeries:
        """ TimeAtlas TimeSeries to Unit8 TimeSeries
        conversion method

        Returns: Unit 8 TimeSeries object
        """
        return U8TimeSeries.from_times_and_values(self.series.index,
                                                  self.series.values)

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
