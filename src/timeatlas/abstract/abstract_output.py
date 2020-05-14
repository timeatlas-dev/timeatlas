from abc import ABC, abstractmethod
from typing import NoReturn


class AbstractOutputText(ABC):
    """ Definition of method signatures for
    data output in text format from time atlas objects
    """

    @abstractmethod
    def to_text(self, path: str, name: str) -> NoReturn:
        raise NotImplementedError


class AbstractOutputPickle(ABC):
    """ Definition of method signatures for
    data output in Pickle format from time atlas objects
    """

    @abstractmethod
    def to_pickle(self, path: str, name: str) -> NoReturn:
        raise NotImplementedError


class AbstractOutputJson(ABC):
    """ Definition of method signatures for
        data output in JSON format from time atlas objects
    """

    @abstractmethod
    def to_json(self) -> NoReturn:
        raise NotImplementedError
