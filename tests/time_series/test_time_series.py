from unittest import TestCase
from pandas import DatetimeIndex, Series

from timeatlas import TimeSeries, MetadataIO


class TestTimeSeries(TestCase):

    def setUp(self) -> None:
        self.my_time_series = TimeSeries()

    def test__TimeSeries__is_instance(self):
        self.assertIsInstance(self.my_time_series, TimeSeries,
                              "The TimeSeries hasn't the right type")

    def test__TimeSeries__has_right_types(self):
        # Add some data
        index = DatetimeIndex(['2019-01-01', '2019-01-02',
                               '2019-01-03', '2019-01-04'])
        my_series = Series([0.4, 1.0, 0.7, 0.6], index=index)
        my_metadata = MetadataIO()
        my_ts = TimeSeries(my_series, my_metadata)
        # Check types
        self.assertIsInstance(my_ts.series, Series,
                              "The TimeSeries Series is not a Pandas Series")
        self.assertIsInstance(my_ts.metadata, MetadataIO,
                              "The TimeSeries Metadata hasn't the right type")

    def test__TimeSeries__wrong_index_type(self):
        values = Series([0.4, 1.0, 0.7, 0.6])
        with self.assertRaises(AssertionError):
            TimeSeries(values)

    def tearDown(self) -> None:
        del self.my_time_series
