from abc import ABC, abstractmethod
from typing import NoReturn


class AbstractBaseManipulator(ABC):
    """ Abstract class to define methods to implement
        for a Manipulator class.
        """

    @abstractmethod
    def extract(self) -> NoReturn:
        raise NotImplementedError
