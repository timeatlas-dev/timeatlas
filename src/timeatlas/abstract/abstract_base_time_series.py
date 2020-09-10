from abc import ABC, abstractmethod
from typing import Any
from pandas import DataFrame


class AbstractBaseTimeSeries(ABC):
    """ Definition of the methods signatures
    usable for Time Series
    """

    # ==========================================================================
    # Methods
    # ==========================================================================

    @abstractmethod
    def create(self, *args) -> Any:
        raise NotImplementedError




    # ==========================================================================
    # Processing
    # ==========================================================================

    @abstractmethod
    def resample(self, by: str) -> Any:
        raise NotImplementedError

    @abstractmethod
    def interpolate(self, method: str) -> Any:
        raise NotImplementedError

    @abstractmethod
    def normalize(self, method: str) -> Any:
        raise NotImplementedError

    # ==========================================================================
    # Analysis
    # ==========================================================================

    @abstractmethod
    def describe(self) -> DataFrame:
        raise NotImplementedError

    @abstractmethod
    def frequency(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def resolution(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def duration(self) -> Any:
        raise NotImplementedError
