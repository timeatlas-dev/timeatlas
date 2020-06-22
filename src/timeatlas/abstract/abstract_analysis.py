from abc import ABC, abstractmethod
from typing import Any
from pandas import DataFrame


class AbstractAnalysis(ABC):
    """ Definition of the methods signatures
    usable for analysis
    """

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
