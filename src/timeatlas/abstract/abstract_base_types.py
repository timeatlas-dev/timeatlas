from abc import ABC, abstractmethod
from typing import List, Tuple


class AbstractBaseMetadataType(ABC):
    """ Definition of the methods signatures
    usable for Metadata Types
    """

    @abstractmethod
    def items(self) -> List[Tuple]:
        """returning a List(Tuples) -> [name, self]"""
        raise NotImplementedError
