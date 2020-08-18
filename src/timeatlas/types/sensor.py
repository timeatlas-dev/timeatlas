
class Sensor:
    """ Defines a sensor """

    def __init__(self, id: int, name: str):
        self.id = id
        self.name = name

    def __repr__(self):
        return "Sensor ID {}".format(self.id)
