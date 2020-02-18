from abc import ABC, abstractmethod
from typing import Any, NoReturn


class AbstractIO(ABC):

    @abstractmethod
    def read(self, path: str) -> Any:
        raise NotImplementedError

    @abstractmethod
    def write(self, path: str, name: str) -> NoReturn:
        raise NotImplementedError
