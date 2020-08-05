
class Sensor:
    """ Defines a sensor """

    def __init__(self, sensor_id: int, name: str):
        self.sensor_id = sensor_id
        self.name = name

    def __repr__(self):
        return "Sensor ID {}".format(self.sensor_id)
