from pandas import Series, DataFrame, date_range, infer_freq
from pandas.plotting import register_matplotlib_converters
from pandas.api.types import infer_dtype
from typing import NoReturn, Tuple, Any
from datetime import datetime

from u8timeseries import TimeSeries as U8TimeSeries

from pandas import Series, DatetimeIndex
from timeatlas.abstract.abstract_analysis import AbstractAnalysis
from timeatlas.abstract import AbstractOutputText, AbstractOutputPickle
from timeatlas.abstract.abstract_processing import AbstractProcessing
from timeatlas.config.constants import (
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
        series: An optional Pandas Series
        metadata: An optional Dict storing metadata about this TimeSeries
    """
    def __init__(self, series: Series = None, metadata: Metadata = None):

        if series is not None:
            # Check if values have a DatetimeIndex
            assert isinstance(series.index, DatetimeIndex), \
                'Values must be indexed with a DatetimeIndex.'

            # Check if the length is bigger than one
            assert len(series) >= 1, 'Values must have at least one values.'

            # Give a default name to the series (for the CSV output)
            series.name = "values"

        # Create the TimeSeries object
        self.series = series

        if metadata is not None:
            self.metadata = metadata
        else:
            self.metadata = None

    # Methods
    # =======

    @staticmethod
    def create(start: str, end: str, freq=None, metadata: Metadata = None):
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
        series = Series(index=date_range(start, end, freq=freq))
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
        before = TimeSeries(self.series[start:splitting_point], self.metadata)
        after = TimeSeries(self.series[splitting_point:end], self.metadata)
        return before, after

    def erase(self):
        pass

    # =============================================
    # Analysis
    # =============================================
    def plot(self):
        """
        Plot a TimeSeries

        Returns:
        """
        register_matplotlib_converters()
        self.series.plot()

    def describe(self):
        """
        Describe a TimeSeries with the describe function from Pandas

        Returns:
            TODO
        """
        raise NotImplementedError
        pass

    def compute_resolution(self) -> 'TimeSeries':
        """
        Create the series of delta T between each timestamp of a TimeSeries
        TODO Should it returns a Series of TimeDelta?

        Returns:
            TODO
        """
        raise NotImplementedError
        pass

    def compute_duration(self) -> Tuple[datetime, datetime]:
        """
        Compute the duration of a TimeSeries by giving you its start and ending
        timestamps in a Tuple

        Returns:
            TODO
        """
        raise NotImplementedError
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
        Intelligent interpolation in function of the data unit etc.
        """
        pass

    def normalize(self, method: str) -> Any:
        """
        Normalize a dataset
        """
        pass

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
        return U8TimeSeries.from_times_and_values(self.series.index, self.series.values)

    @staticmethod
    def from_df(df: DataFrame, values_column: str, index_column: str = None) -> 'TimeSeries':
        """Pandas DataFrame to TimeAtlas TimeSeries
        conversion method

        Returns: TimeAtlas TimeSeries
        """
        series = Series(data=df[values_column].values) \
            if index_column is None \
            else Series(data=df[values_column].values, index=df[index_column])
        return TimeSeries(series)

    def to_df(self) -> DataFrame:
        """ TimeAtlas TimeSeries to Pandas DataFrame
        conversion method

        Returns: Pandas DataFrame
        """
        return DataFrame(self.series.values,
                         index=self.series.index,
                         columns=["values"])

    @staticmethod
    def __series_to_csv(series: Series, path: str):
        """
        Read a Pandas Series and put it into a CSV file

        Args:
            series: The Series to write in CSV
            path: The path where the Series will be saved
        """
        series.to_csv(path, header=True, index_label="index")
