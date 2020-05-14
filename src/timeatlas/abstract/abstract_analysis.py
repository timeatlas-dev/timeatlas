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
    def compute_resolution(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def compute_duration(self) -> Any:
        raise NotImplementedError
