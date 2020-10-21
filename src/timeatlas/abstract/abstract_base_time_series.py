from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Union, Tuple, List, Callable, Optional, \
    NoReturn
from pandas import DataFrame, Timestamp, Timedelta

if TYPE_CHECKING:
    from timeatlas.time_series import TimeSeries
    from timeatlas.time_series_dataset import TimeSeriesDataset
    from timeatlas.abstract.abstract_base_model import AbstractBaseModel


class AbstractBaseTimeSeries(ABC):
    """ Definition of the methods signatures
    usable for Time Series
    """

    # ==========================================================================
    # Methods
    # ==========================================================================

    @abstractmethod
    def create(self, *args) -> Union['TimeSeries', 'TimeSeriesDataset']:
        raise NotImplementedError

    @abstractmethod
    def plot(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    @abstractmethod
    def split_at(self, timestamp: Union[str, Timestamp]) \
            -> Union[Tuple['TimeSeries', 'TimeSeries'],
                     Tuple['TimeSeriesDataset', 'TimeSeriesDataset']]:
        raise NotImplementedError

    @abstractmethod
    def split_in_chunks(self, n: int) \
            -> Union[List['TimeSeries'], List['TimeSeriesDataset']]:
        raise NotImplementedError

    @abstractmethod
    def fill(self, value: Any) -> Union['TimeSeries', 'TimeSeriesDataset']:
        raise NotImplementedError

    @abstractmethod
    def empty(self) -> Union['TimeSeries', 'TimeSeriesDataset']:
        raise NotImplementedError

    @abstractmethod
    def pad(self, *args) -> Union['TimeSeries', 'TimeSeriesDataset']:
        raise NotImplementedError

    @abstractmethod
    def trim(self, side: str) -> Union['TimeSeries', 'TimeSeriesDataset']:
        raise NotImplementedError

    @abstractmethod
    def merge(self, ts: Union['TimeSeries', 'TimeSeriesDataset']) \
            -> Union['TimeSeries', 'TimeSeriesDataset']:
        raise NotImplementedError

    # ==========================================================================
    # Processing
    # ==========================================================================

    @abstractmethod
    def apply(self, func, ts: Union['TimeSeries', 'TimeSeriesDataset'] = None) \
            -> Union['TimeSeries', 'TimeSeriesDataset']:
        raise NotImplementedError

    @abstractmethod
    def resample(self, freq: Union[str],
                 method: Optional[Union[str, 'AbstractBaseModel']] = None) \
            -> Union['TimeSeries', 'TimeSeriesDataset']:
        raise NotImplementedError

    @abstractmethod
    def interpolate(self, method: str) \
            -> Union['TimeSeries', 'TimeSeriesDataset']:
        raise NotImplementedError

    @abstractmethod
    def normalize(self, method: str) \
            -> Union['TimeSeries', 'TimeSeriesDataset']:
        raise NotImplementedError

    @abstractmethod
    def round(self, decimals: int) -> Union['TimeSeries', 'TimeSeriesDataset']:
        raise NotImplementedError

    @abstractmethod
    def sort(self, *args) -> Union['TimeSeries', 'TimeSeriesDataset']:
        raise NotImplementedError

    # ==========================================================================
    # Analysis
    # ==========================================================================

    # Basic Statistics
    # ----------------

    @abstractmethod
    def min(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def max(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def mean(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def median(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def kurtosis(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def skewness(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def describe(self) -> DataFrame:
        raise NotImplementedError

    # Time Series Statistics
    # ----------------------

    @abstractmethod
    def start(self) -> Union[Timestamp, List[Timestamp]]:
        raise NotImplementedError

    @abstractmethod
    def end(self) -> Union[Timestamp, List[Timestamp]]:
        raise NotImplementedError

    @abstractmethod
    def boundaries(self) \
            -> Union[Tuple[Timestamp, Timestamp],
                     List[Tuple[Timestamp, Timestamp]]]:
        raise NotImplementedError

    @abstractmethod
    def frequency(self) -> Union[Optional[str], List[Optional[str]]]:
        raise NotImplementedError

    @abstractmethod
    def time_detlas(self) -> Union['TimeSeries', 'TimeSeriesDataset']:
        raise NotImplementedError

    @abstractmethod
    def duration(self) -> Union[Timedelta, List[Timedelta]]:
        raise NotImplementedError
