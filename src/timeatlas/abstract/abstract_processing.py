from abc import ABC, abstractmethod
from typing import Any


class AbstractProcessing(ABC):

    @abstractmethod
    def resample(self, by: str) -> Any:
        raise NotImplementedError

    @abstractmethod
    def interpolate(self, method: str) -> Any:
        raise NotImplementedError

    @abstractmethod
    def normalize(self, method: str) -> Any:
        raise NotImplementedError
