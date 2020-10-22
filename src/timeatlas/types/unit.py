from timeatlas.abstract import AbstractBaseMetadataType


class Unit(AbstractBaseMetadataType):
    """ Defines a physical unit of measurement, like Celsius."""

    def __init__(self, name: str, symbol: str, data_type: str):
        self.name = name
        self.symbol = symbol
        self.data_type = data_type

    def items(self):
        return [("unit", self)]
