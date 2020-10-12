from abc import ABC, abstractmethod
from typing import Any


class AbstractProcessing(ABC):
    """ Definition of method signatures for
    processing on time atlas objects
    """

    @abstractmethod
    def resample(self, by: str) -> Any:
        raise NotImplementedError

    @abstractmethod
    def interpolate(self, method: str) -> Any:
        raise NotImplementedError

    @abstractmethod
    def normalize(self, method: str) -> Any:
        raise NotImplementedError
