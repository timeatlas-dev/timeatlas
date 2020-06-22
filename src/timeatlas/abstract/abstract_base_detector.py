from abc import ABC, abstractmethod
from typing import NoReturn


class AbstractBaseDetector(ABC):
    """ Abstract class to define methods to implement
    for a Detector class.
    """

    @abstractmethod
    def detect(self, ts) -> NoReturn:
        """ Detect features """
        raise NotImplementedError
