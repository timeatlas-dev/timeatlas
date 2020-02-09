from dataclasses import dataclass


@dataclass
class Sensor:
    """ Represents the data source of a time series data

    The Sensor object aims at representing a physical or virtual device
    issuing the time series data.

    """
    id: int
    name: str
