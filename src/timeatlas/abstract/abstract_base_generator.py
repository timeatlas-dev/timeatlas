from abc import ABC, abstractmethod
from typing import NoReturn


class AbstractBaseGenerator(ABC):
    """ Abstract class to define methods to implement
    for a Generator class.
    """

    @abstractmethod
    def __init__(self):
        # Each generator should be called the super class and set a label_suffix
        self.label_suffix = None

    @abstractmethod
    def generate(self) -> NoReturn:
        """ Generate features """
        raise NotImplementedError
