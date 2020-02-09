from unittest import TestCase
from pandas import DatetimeIndex, Series

from timeatlas.data_structures import Unit, Sensor, TimeSeries


class TestTimeSeries(TestCase):

    def setUp(self) -> None:
        self.my_sensor = Sensor(1, "My weather sensor")
        self.my_unit = Unit("Temperature", "°C", "float")
        index = DatetimeIndex(['2019-01-01', '2019-01-02',
                               '2019-01-03', '2019-01-04'])
        self.my_series = Series([0.4, 1.0, 0.7, 0.6], index=index)
        self.my_time_series = TimeSeries(self.my_series, self.my_sensor,
                                         self.my_unit)

    def test__TimeSeries__has_right_types(self):
        self.assertTrue(type(self.my_time_series._unit) is Unit)
        self.assertTrue(type(self.my_time_series._sensor) is Sensor)
        self.assertTrue(type(self.my_time_series._series) is Series)

    def test__TimeSeries__is_instance(self):
        self.assertIsInstance(self.my_time_series, TimeSeries)

    def test__TimeSeries__wrong_index_type(self):
        values = Series([0.4, 1.0, 0.7, 0.6])
        with self.assertRaises(AssertionError):
            TimeSeries(values, self.my_unit, self.my_sensor)
