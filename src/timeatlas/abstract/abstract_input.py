from abc import ABC, abstractmethod
from typing import Any


class AbstractInput(ABC):
    """ Definition of method signatures for
    data input into time atlas objects
    """

    @abstractmethod
    def read(self, path: str) -> Any:
        raise NotImplementedError
