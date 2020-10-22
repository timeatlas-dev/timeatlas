from typing import List, Tuple

from timeatlas.abstract import AbstractBaseMetadataType


class Coords(AbstractBaseMetadataType):
    """ Defines geographic coordinates

    The format to use is decimal degrees (DD). For instance:

        46.9491463,7.4388499

    For the train station of Bern, the capital of Switzerland.
    """

    def __init__(self, lat: float, long: float):
        self.lat = lat
        self.long = long

    def __repr__(self):
        return f"{self.lat}°N, {self.long}°E"

    def items(self) -> List[Tuple]:
        """Creating dict.iterable

        Imitating the dict iterable

        for k, v in dict.items()

        Returns:
            List[Tuple]

        """
        return [("coords", self)]
