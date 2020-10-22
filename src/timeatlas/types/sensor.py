from timeatlas.abstract import AbstractBaseMetadataType


class Sensor(AbstractBaseMetadataType):
    """ Defines a sensor """

    def __init__(self, id: int, name: str):
        self.id = id
        self.name = name

    def __repr__(self):
        return f"Sensor ID: {self.id}; Name: {self.name}"

    def items(self):
        return [("sensor", self)]
