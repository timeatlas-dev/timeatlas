from abc import ABC, abstractmethod
from typing import Any, NoReturn


class AbstractInput(ABC):
    """ Definition of method signatures for
    data input into time atlas objects
    """

    @abstractmethod
    def read(self, path: str) -> Any:
        raise NotImplementedError


class AbstractOutputText(ABC):
    """ Definition of method signatures for
    data output in text format from time atlas objects
    """

    @abstractmethod
    def to_text(self, path: str) -> NoReturn:
        raise NotImplementedError


class AbstractOutputPickle(ABC):
    """ Definition of method signatures for
    data output in Pickle format from time atlas objects
    """

    @abstractmethod
    def to_pickle(self, path: str) -> NoReturn:
        raise NotImplementedError


class AbstractOutputJson(ABC):
    """ Definition of method signatures for
        data output in JSON format from time atlas objects
    """

    @abstractmethod
    def to_json(self, path: str) -> NoReturn:
        raise NotImplementedError
