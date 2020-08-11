from unittest import TestCase
from pandas import DatetimeIndex, DataFrame, Series
from timeatlas import TimeSeries, Metadata
from timeatlas.config.constants import TIME_SERIES_VALUES


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
        my_series = DataFrame([0.4, 1.0, 0.7, 0.6], index=index)
        my_metadata = Metadata()
        my_ts = TimeSeries(my_series, my_metadata)
        # Check types
        self.assertIsInstance(my_ts.series, DataFrame,
                              "The TimeSeries series is not a Pandas DataFrame")
        self.assertIsInstance(my_ts.metadata, Metadata,
                              "The TimeSeries Metadata hasn't got the right type")

    def test__TimeSeries__wrong_index_type(self):
        values = Series([0.4, 1.0, 0.7, 0.6])
        with self.assertRaises(AssertionError):
            TimeSeries(values)

    def test__TimeSeries__with_Series_input(self):
        index = DatetimeIndex(['2019-01-01', '2019-01-02', '2019-01-03', '2019-01-04'])
        my_series = Series([0.4, 1.0, 0.7, 0.6], index=index)
        ts = TimeSeries(my_series)
        self.assertTrue(TIME_SERIES_VALUES in ts.series.columns)
        self.assertIsInstance(ts, TimeSeries)


    def test__TimeSeries__with_DataFrame_input_single_column(self):
        index = DatetimeIndex(['2019-01-01', '2019-01-02', '2019-01-03', '2019-01-04'])
        my_series = Series([0.4, 1.0, 0.7, 0.6], index=index)
        df = DataFrame(data=my_series)
        ts = TimeSeries(df)
        self.assertTrue(TIME_SERIES_VALUES in ts.series.columns)
        self.assertIsInstance(ts, TimeSeries)

    def test__TimeSeries__with_DataFrame_input_many_columns__without_values(self):
        index = DatetimeIndex(['2019-01-01', '2019-01-02', '2019-01-03', '2019-01-04'])
        my_series = Series([0.4, 1.0, 0.7, 0.6], index=index)
        df = DataFrame({"one": my_series, "two": my_series})
        with self.assertRaises(AssertionError):
            ts = TimeSeries(df)

    def test__TimeSeries__with_DataFrame_input_many_columns__with_values(self):
        index = DatetimeIndex(['2019-01-01', '2019-01-02', '2019-01-03', '2019-01-04'])
        my_series = Series([0.4, 1.0, 0.7, 0.6], index=index)
        df = DataFrame({TIME_SERIES_VALUES: my_series, "two": my_series})
        ts = TimeSeries(df)
        self.assertIsInstance(ts, TimeSeries)

    def tearDown(self) -> None:
        del self.my_time_series
