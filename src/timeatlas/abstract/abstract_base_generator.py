from abc import ABC, abstractmethod
from typing import NoReturn


class AbstractBaseGenerator(ABC):
    """ Abstract class to define methods to implement
    for a Generator class.
    """

    @abstractmethod
    def generate(self) -> NoReturn:
        """ Generate features """
        raise NotImplementedError
