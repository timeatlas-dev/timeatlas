from dataclasses import dataclass
from pandas import Series, DatetimeIndex

from .sensor import Sensor
from .unit import Unit


@dataclass
class TimeSeries:
    """ Defines a time series

    A TimeSeries object is an immutable series of time indexed values. It has
    two optional parameters, unit and sensor.

    As its name suggest, Unit represents the unit of the values from the time
    series.

    Sensor represents the physical or virtual data source of the time series.
    """

    def __init__(self, series: Series, unit: Unit = None,
                 sensor: Sensor = None,):

        # Check if values have a DatetimeIndex
        assert isinstance(series.index, DatetimeIndex), \
            'Values must be indexed with a DatetimeIndex.'

        # Check if the length is bigger than one
        assert len(series) >= 1, 'Values must have at least one values.'

        # Create the TimeSeries object
        self._series = series
        self._sensor = sensor
        self._unit = unit



