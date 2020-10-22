from abc import ABC, abstractmethod
from typing import List, Tuple


class AbstractBaseMetadataType(ABC):

    @abstractmethod
    def items(self) -> List[Tuple]:
        raise NotImplementedError
