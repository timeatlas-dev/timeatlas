from unittest import TestCase
from pandas import DatetimeIndex, DataFrame, Series
import numpy as np
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

    def test__TimeSeries__create(self):
        ts = TimeSeries.create("01-01-2020", "02-01-2020")
        self.assertIsInstance(ts, TimeSeries)
        # Test if all elements are NaNs
        self.assertTrue(ts.series.isna().all().values[0])
        # Test if frequency is daily
        self.assertEqual(ts.series.index.inferred_freq, 'D')

    def test__TimeSeries__create_with_freq_as_str(self):
        ts = TimeSeries.create("01-01-2020", "02-01-2020", "H")
        self.assertIsInstance(ts, TimeSeries)
        # Test if all elements are NaNs
        self.assertTrue(ts.series.isna().all().values[0])
        # Test if frequency is daily
        self.assertEqual(ts.series.index.inferred_freq, 'H')

    def test__TimeSeries__create_with_freq_as_time_series(self):
        ts_freq = TimeSeries.create("01-01-2020", "02-01-2020", "H")
        ts = TimeSeries.create("01-01-2020", "02-01-2020", ts_freq)
        self.assertIsInstance(ts, TimeSeries)
        # Test if all elements are NaNs
        self.assertTrue(ts.series.isna().all().values[0])
        # Test if frequency is daily
        self.assertEqual(ts.series.index.inferred_freq,
                         ts_freq.series.index.inferred_freq)

    def test__TimeSeries__register_plotting_function(self):
        # TODO
        pass

    def test__TimeSeries__plot(self):
        # TODO
        pass

    def test__TimeSeries__split_at(self):
        # Create TimeSeries and split it
        ts = TimeSeries.create("01-01-2020", "03-01-2020", "H")
        a, b = ts.split_at("02-01-2020 00:00")
        # Get all the indexes
        ts_start = ts.series["values"].index[0]
        ts_end = ts.series["values"].index[-1]
        a_start = a.series["values"].index[0]
        a_end = a.series["values"].index[-1]
        b_start = b.series["values"].index[0]
        b_end = b.series["values"].index[-1]
        # Test boundaries
        self.assertEqual(ts_start, a_start)
        self.assertEqual(ts_end, b_end)
        # Test split point
        self.assertEqual(a_end, b_start)

    def test__TimeSeries__split_in_chunks(self):
        # TODO
        pass

    def test__TimeSeries__fill(self):
        # TODO
        pass

    def test__TimeSeries__empty(self):
        # TODO
        pass

    def test__TimeSeries__trim(self):
        # TODO
        pass

    def test__TimeSeries__merge(self):
        # Prepare test
        ts1 = TimeSeries.create("01-2020", "03-2020", "H")
        ts2 = TimeSeries.create("02-2020", "04-2020", "H")
        # Call function
        ts = ts1.merge(ts2)
        # Test if index is monotonic increasing
        self.assertTrue(ts.series.index.is_monotonic_increasing)
        # Test if all values are there
        len1 = len(ts1) + len(ts2)
        len2 = len(ts)
        self.assertTrue(len1 == len2)

    def test__TimeSeries__to_darts__type_check(self):
        ts = TimeSeries.create("01-2020", "02-2020", "H")
        ts = ts.fill(np.random.randint(0,1000,len(ts)))
        self.assertIsInstance(ts, TimeSeries)
        dts = ts.to_darts()
        from darts import TimeSeries as DartsTimeSeries
        self.assertIsInstance(dts, DartsTimeSeries)

    def test__TimeSeries__to_darts__series_equality(self):
        ts = TimeSeries.create("01-2020", "02-2020", "H")
        ts = ts.fill(np.random.randint(0,1000,len(ts)))
        dts = ts.to_darts()
        is_equal = ts.series[TIME_SERIES_VALUES].equals(dts.pd_series())
        self.assertTrue(is_equal)

    def tearDown(self) -> None:
        del self.my_time_series
