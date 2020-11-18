from unittest import TestCase
from os import path as os_path
import shutil

from pandas import DatetimeIndex, DataFrame, Series, Timedelta, infer_freq, \
    Timestamp
import numpy as np
from plotly.graph_objects import Figure
from matplotlib.axes import Axes

from timeatlas import TimeSeries, Metadata
from timeatlas.config.constants import *


class TestTimeSeries(TestCase):

    def setUp(self) -> None:

        # Create a time indexed series
        index = DatetimeIndex(['2019-01-01', '2019-01-02',
                               '2019-01-03', '2019-01-04'])
        self.my_data = Series([0.4, 1.0, 0.7, 0.6], index=index).to_frame()

        # Create metadata
        my_unit = {
            "name": "power",
            "symbol": "W",
            "data_type": "float"
        }
        my_coordinates = {
            "lat": 46.796611,
            "lon": 7.147563
        }
        my_dict = {
            "unit": my_unit,
            "coordinates": my_coordinates
        }
        self.my_metadata = Metadata(my_dict)

        #self.my_time_series = TimeSeries(self.my_series, self.my_metadata)

        self.my_time_series = TimeSeries(self.my_data)

        # Define a target directory
        self.target_dir = "data/test-export"

    def test__init__is_instance(self):
        self.my_time_series = TimeSeries()
        self.assertIsInstance(self.my_time_series, TimeSeries,
                              "The TimeSeries hasn't the right type")

    def test__init__has_right_types(self):
        # Add some data
        index = DatetimeIndex(['2019-01-01', '2019-01-02',
                               '2019-01-03', '2019-01-04'])
        my_series = DataFrame([0.4, 1.0, 0.7, 0.6], index=index)
        my_metadata = Metadata()
        my_ts = TimeSeries(my_series, my_metadata)
        # Check types
        self.assertIsInstance(my_ts.data, DataFrame,
                              "The TimeSeries series is not a Pandas DataFrame")
        self.assertIsInstance(my_ts.metadata, Metadata,
                              "The TimeSeries Metadata hasn't got the right type")

    def test__init__contains_metadata(self):
        # Add some data
        index = DatetimeIndex(['2019-01-01', '2019-01-02',
                               '2019-01-03', '2019-01-04'])
        my_series = DataFrame([0.4, 1.0, 0.7, 0.6], index=index)
        my_metadata = Metadata()
        my_ts = TimeSeries(my_series, my_metadata)
        # Check types
        self.assertNotEqual(my_ts.metadata, None,
                            "The TimeSeries Metadata is probably None")

    def test__init__has_values_as_column_name(self):
        index = DatetimeIndex(['2019-01-01', '2019-01-02', '2019-01-03', '2019-01-04'])
        my_series = Series([0.4, 1.0, 0.7, 0.6], index=index)
        ts = TimeSeries(my_series)
        self.assertTrue(TIME_SERIES_VALUES in ts.data.columns)

    def test__init__wrong_index_type(self):
        values = Series([0.4, 1.0, 0.7, 0.6])
        with self.assertRaises(AssertionError):
            TimeSeries(values)

    def test__init__with_Series_input(self):
        index = DatetimeIndex(['2019-01-01', '2019-01-02', '2019-01-03', '2019-01-04'])
        my_series = Series([0.4, 1.0, 0.7, 0.6], index=index)
        ts = TimeSeries(my_series)
        self.assertTrue(TIME_SERIES_VALUES in ts.data.columns)
        self.assertIsInstance(ts, TimeSeries)

    def test__init__with_DataFrame_input_single_column(self):
        index = DatetimeIndex(['2019-01-01', '2019-01-02', '2019-01-03', '2019-01-04'])
        my_series = Series([0.4, 1.0, 0.7, 0.6], index=index)
        df = DataFrame(data=my_series)
        ts = TimeSeries(df)
        self.assertTrue(TIME_SERIES_VALUES in ts.data.columns)
        self.assertIsInstance(ts, TimeSeries)

    def test__init__with_DataFrame_input_many_columns__without_values(self):
        index = DatetimeIndex(['2019-01-01', '2019-01-02', '2019-01-03', '2019-01-04'])
        my_series = Series([0.4, 1.0, 0.7, 0.6], index=index)
        df = DataFrame({"one": my_series, "two": my_series})
        with self.assertRaises(AssertionError):
            ts = TimeSeries(df)

    def test__init__with_DataFrame_input_many_columns__with_values(self):
        index = DatetimeIndex(['2019-01-01', '2019-01-02', '2019-01-03', '2019-01-04'])
        my_series = Series([0.4, 1.0, 0.7, 0.6], index=index)
        df = DataFrame({TIME_SERIES_VALUES: my_series, "two": my_series})
        ts = TimeSeries(df)
        self.assertIsInstance(ts, TimeSeries)

    def test__init__freq_is_infered(self):
        index = DatetimeIndex(['2019-01-01', '2019-01-02', '2019-01-03', '2019-01-04'])
        my_series = Series([0.4, 1.0, 0.7, 0.6], index=index)
        ts = TimeSeries(my_series)
        self.assertEqual(infer_freq(index), ts.data.index.freq)

    def test__getitem__with_int(self):
        # object
        ts = TimeSeries.create("01-01-2020", "01-03-2020", "H").fill(0)
        # test selection
        expected = ts.data.iloc[[0]]
        actual = ts[0].data
        self.assertTrue(actual.equals(expected))
        # test handler similarity
        expected_handler = ts.handler
        actual_handler = ts[0].handler
        self.assertEqual(id(actual_handler), id(expected_handler))

    def test__getitem__with_str__select_the_right_data(self):
        # object
        ts1 = TimeSeries.create("01-01-2020", "01-03-2020", "H") .fill(0)
        ts2 = TimeSeries.create("01-01-2020", "01-03-2020", "H") .fill(1)
        ts = ts1.stack(ts2)
        # test data
        expected = ts.data.loc[:, ["0_values"]]
        actual = ts["0_values"].data
        self.assertTrue(actual.equals(expected))

    def test__getitem__with_str__select_the_right_component_in_handler(self):
        # object
        ts1 = TimeSeries.create("01-01-2020", "01-03-2020", "H") .fill(0)
        ts2 = TimeSeries.create("01-01-2020", "01-03-2020", "H") .fill(1)
        ts = ts1.stack(ts2)
        # test presence of two components in handler
        self.assertEqual(ts.handler.get_columns().to_list(),
                         ["0_values", "1_values"])
        # test that selection with one component returns one component in
        # handler
        new_ts = ts["0_values"]
        self.assertEqual(new_ts.handler.get_columns().to_list(),
                         ["0_values"])
        nb_components = len(new_ts.handler.components)
        self.assertEqual(nb_components, 1)
        # test that it is the right component
        c = new_ts.handler.get_component_by_name("0_values")
        self.assertEqual(c.get_main(), "values")

    def test__getitem__with_timestamp(self):
        # object
        ts = TimeSeries.create("01-01-2020", "01-03-2020", "H").fill(0)
        timestamp = Timestamp("01-01-2020")
        # test selection
        expected = ts.data.loc[[timestamp]]
        actual = ts[timestamp].data
        self.assertTrue(actual.equals(expected))
        # test handler similarity
        expected_handler = ts.handler
        actual_handler = ts[timestamp].handler
        self.assertEqual(id(actual_handler), id(expected_handler))

    def test__getitem__with_slice_of_int(self):
        # object
        ts = TimeSeries.create("01-01-2020", "01-03-2020", "H").fill(0)
        # test selection
        expected = ts.data.iloc[0:5]
        actual = ts[0:5].data
        self.assertTrue(actual.equals(expected))
        # test handler similarity
        expected_handler = ts.handler
        actual_handler = ts[0:5].handler
        self.assertEqual(id(actual_handler), id(expected_handler))

    def test__getitem__with_slice_of_str(self):
        # object
        ts = TimeSeries.create("01-01-2020", "01-03-2020", "H").fill(0)
        # test selection
        expected = ts.data.loc["01-01-2020 10:00":"01-01-2020 20:00"]
        actual = ts["01-01-2020 10:00":"01-01-2020 20:00"].data
        self.assertTrue(actual.equals(expected))
        # test handler similarity
        expected_handler = ts.handler
        actual_handler = ts["01-01-2020 10:00":"01-01-2020 20:00"].handler
        self.assertEqual(id(actual_handler), id(expected_handler))

    def test__getitem__with_list_of_int(self):
        # object
        ts1 = TimeSeries.create("01-01-2020", "01-03-2020", "H").fill(0)
        ts2 = TimeSeries.create("01-01-2020", "01-03-2020", "H").fill(1)
        ts3 = TimeSeries.create("01-01-2020", "01-03-2020", "H").fill(2)
        ts4 = TimeSeries.create("01-01-2020", "01-03-2020", "H").fill(3)
        ts = ts1.stack(ts2).stack(ts3).stack(ts4)
        # function at test
        selection = [0, 1, 3]
        new_ts = ts[selection]
        # test selection
        expected = ts.data.iloc[:, selection]
        expected = expected.rename(columns={"3_values": "2_values"})
        actual = new_ts.data
        self.assertTrue(actual.equals(expected))
        # test handler similarity
        self.assertEqual(new_ts.handler.get_columns().to_list(),
                         ["0_values", "1_values", "2_values"])
        # test nb of components
        nb_components = len(new_ts.handler.components)
        self.assertEqual(nb_components, 3)

    def test__getitem__with_list_of_str(self):
        # object
        ts1 = TimeSeries.create("01-01-2020", "01-03-2020", "H").fill(0)
        ts2 = TimeSeries.create("01-01-2020", "01-03-2020", "H").fill(1)
        ts3 = TimeSeries.create("01-01-2020", "01-03-2020", "H").fill(2)
        ts4 = TimeSeries.create("01-01-2020", "01-03-2020", "H").fill(3)
        ts = ts1.stack(ts2).stack(ts3).stack(ts4)
        # function at test
        selection = ["0_values", "2_values"]
        new_ts = ts[selection]
        # test selection
        expected = ts.data.loc[:, selection]
        expected = expected.rename(columns={"2_values": "1_values"})
        actual = new_ts.data
        self.assertTrue(actual.equals(expected))
        # test handler similarity
        self.assertEqual(new_ts.handler.get_columns().to_list(),
                         ["0_values", "1_values"])
        # test nb of components
        nb_components = len(new_ts.handler.components)
        self.assertEqual(nb_components, 2)

    def test__create(self):
        ts = TimeSeries.create("01-01-2020", "02-01-2020")
        self.assertIsInstance(ts, TimeSeries)
        # Test if all elements are NaNs
        self.assertTrue(ts.data.isna().all().values[0])
        # Test if frequency is daily
        self.assertEqual(ts.data.index.inferred_freq, 'D')

    def test__create__is_regular(self):
        ts = TimeSeries.create("01-01-2020", "02-01-2020")
        no_duration_diff = ts.index.to_series().__list_diff().__list_diff()[2:] == \
                           Timedelta(0)
        is_regular = no_duration_diff.eq(True).all()
        self.assertTrue(is_regular)

    def test__create__with_freq_as_str(self):
        ts = TimeSeries.create("01-01-2020", "02-01-2020", "H")
        self.assertIsInstance(ts, TimeSeries)
        # Test if all elements are NaNs
        self.assertTrue(ts.data.isna().all().values[0])
        # Test if frequency is daily
        self.assertEqual(ts.data.index.inferred_freq, 'H')

    def test__create__with_freq_as_time_series(self):
        ts_freq = TimeSeries.create("01-01-2020", "02-01-2020", "H")
        ts = TimeSeries.create("01-01-2020", "02-01-2020", ts_freq)
        self.assertIsInstance(ts, TimeSeries)
        # Test if all elements are NaNs
        self.assertTrue(ts.data.isna().all().values[0])
        # Test if frequency is daily
        self.assertEqual(ts.data.index.inferred_freq,
                         ts_freq.data.index.inferred_freq)

    def test__stack(self):
        # object
        ts1 = TimeSeries.create("01-01-2020", "01-03-2020", "H")
        ts2 = TimeSeries.create("01-01-2020", "01-03-2020", "H") .fill(0)
        # test
        ts3 = ts1.stack(ts2)
        s1 = ts1.data["0_values"]
        s2 = ts2.data["0_values"]
        ss1 = ts3.data["0_values"]
        ss2 = ts3.data["1_values"]
        self.assertTrue(s1.equals(ss1))
        self.assertTrue(s2.equals(ss2))

    def test__drop(self):
        # object
        ts1 = TimeSeries.create("01-01-2020", "01-03-2020", "H")
        ts2 = TimeSeries.create("01-01-2020", "01-03-2020", "H") .fill(0)
        ts3 = ts1.stack(ts2)
        # test
        ts4 = ts3.drop(0)
        s2 = ts2.data["0_values"]
        s4 = ts4.data["0_values"]
        self.assertTrue(s2.equals(s4))

    def test__add_meta(self):
        # object
        ts = TimeSeries.create("01-01-2020", "01-03-2020", "H").fill(2)
        ms1 = TimeSeries.create("01-01-2020", "01-03-2020", "H").fill(1)
        ms2 = TimeSeries.create("01-01-2020", "01-03-2020", "H").fill(3)
        # function at test
        ts = ts.add_meta(ms1, "ci_lower", "0_values")
        ts = ts.add_meta(ms2, "ci_upper", "0_values")
        # test handler contains the right columns
        expected = ["0_values", "0-0_ci_lower", "0-1_ci_upper"]
        actual = ts.handler.get_columns().to_list()
        self.assertEqual(expected, actual)
        # test data has the right infos
        expected = ms1
        expected = expected.data.rename(columns={"0_values": "0-0_ci_lower"})
        actual = ts.data.loc[:, ["0-0_ci_lower"]]
        self.assertTrue(expected.equals(actual))
        # test data has the right columns
        expected = ["0_values", "0-0_ci_lower", "0-1_ci_upper"]
        actual = ts["0_values"].data.columns.to_list()
        self.assertEquals(expected, actual)

    def test__drop_meta__on_one_component(self):
        # object
        ts = TimeSeries.create("01-01-2020", "01-03-2020", "H").fill(2)
        ms1 = TimeSeries.create("01-01-2020", "01-03-2020", "H").fill(1)
        ms2 = TimeSeries.create("01-01-2020", "01-03-2020", "H").fill(3)
        ts = ts.add_meta(ms1, "ci_lower", "0_values")
        ts = ts.add_meta(ms2, "ci_upper", "0_values")
        # function at test
        ts.drop_meta("0_values")

    def test__drop_meta__on_all_component(self):
        # object
        ts = TimeSeries.create("01-01-2020", "01-03-2020", "H").fill(2)
        ms1 = TimeSeries.create("01-01-2020", "01-03-2020", "H").fill(1)
        ms2 = TimeSeries.create("01-01-2020", "01-03-2020", "H").fill(3)
        ts = ts.add_meta(ms1, "ci_lower", "0_values")
        ts = ts.add_meta(ms2, "ci_upper", "0_values")
        # function at test
        ts.drop_meta()

    def test__plot__returns_graph_object_axes(self):
        ts = TimeSeries.create("01-01-2020", "02-01-2020", "H")
        my_fig = ts.plot()
        self.assertIsInstance(my_fig, Axes)

    def test__plot__returns_graph_object_plotly(self):
        ts = TimeSeries.create("01-01-2020", "02-01-2020", "H")
        my_fig = ts.plot(context="notebook")
        self.assertIsInstance(my_fig, Figure)

    def test__copy__shallow(self):
        # object creation
        ts = TimeSeries.create("01-2020", "03-2020", "H")
        copy = ts.copy(deep=False)
        self.assertNotEqual(id(ts), id(copy))

    def test__copy__deep(self):
        # object creation
        ts = TimeSeries.create("01-2020", "03-2020", "H")
        copy = ts.copy(deep=True)
        self.assertNotEqual(id(ts), id(copy))

    def test__split_at(self):
        # Create TimeSeries and split it
        ts = TimeSeries.create("01-01-2020", "03-01-2020", "H")
        a, b = ts.split_at("02-01-2020 00:00")
        # Get all the indexes
        ts_start = ts.data[TIME_SERIES_VALUES].index[0]
        ts_end = ts.data[TIME_SERIES_VALUES].index[-1]
        a_start = a.data[TIME_SERIES_VALUES].index[0]
        a_end = a.data[TIME_SERIES_VALUES].index[-1]
        b_start = b.data[TIME_SERIES_VALUES].index[0]
        b_end = b.data[TIME_SERIES_VALUES].index[-1]
        # Test boundaries
        self.assertEqual(ts_start, a_start)
        self.assertEqual(ts_end, b_end)
        # Test split point
        self.assertEqual(a_end, b_start)

    def test__split_in_chunks(self):
        ts = TimeSeries.create("01-01-2020", "03-01-2020", "H")
        chunk_len = 5
        # method at test
        chunks = ts.split_in_chunks(chunk_len)
        # test all element but last
        for ts_chunk in chunks[:-1]:
            self.assertEqual(len(ts_chunk), chunk_len)
        # test last element
        self.assertLessEqual(len(chunks[-1]), chunk_len)

    def test__fill(self):
        ts = TimeSeries.create("01-01-2020", "03-01-2020", "H")
        val = 42
        # method at test
        ts = ts.fill(val)
        # test
        for i in ts:
            self.assertEqual(val, i)

    def test__empty(self):
        ts = TimeSeries.create("01-01-2020", "03-01-2020", "H")
        # method at test
        ts = ts.empty()
        # test
        for i in ts:
            self.assertTrue(np.isnan(i))

    def test__pad(self):

        def is_regular(ts):
            # test if double timestamps in ts
            no_duration_diff = ts.index.to_series().__list_diff().__list_diff()[2:] == \
                               Timedelta(0)
            return no_duration_diff.eq(True).all()

        def is_monotonic(ts):
            # test if monotonic
            return ts.data.index.is_monotonic

        def is_freq_similar(ts_before, ts_after):
            # test if freq is the same
            return ts_before.frequency() == ts_after.frequency()

        # Create TimeSeries
        ts_1 = TimeSeries.create("04-2020", "05-2020", "D")

        # Pad before
        ts_1_padded_before = ts_1.pad("03-2020")["2020-03-31":"2020-04-02"]
        self.assertTrue(is_regular(ts_1_padded_before))
        self.assertTrue(is_monotonic(ts_1_padded_before))
        self.assertTrue(is_freq_similar(ts_1, ts_1_padded_before))

        # Pad after
        ts_1_padded_after = ts_1.pad("06-2020")["2020-05-29":"2020-06-02"]
        self.assertTrue(is_regular(ts_1_padded_after))
        self.assertTrue(is_monotonic(ts_1_padded_after))
        self.assertTrue(is_freq_similar(ts_1, ts_1_padded_after))

        # Pad during (wrong case)
        with self.assertRaises(ValueError):
            ts_1.pad("2020-04-15")

    def test__trim__both_side_by_default(self):
        # Prepare TimeSeries with int as values
        start = Timestamp("01-01-2020")
        end = Timestamp("03-01-2020")
        ts_initial = TimeSeries.create(start, end, "H")
        ts_initial = ts_initial.fill(42)
        # Pad the TimeSeries with np.nans
        ts_pad = ts_initial.pad("01-12-2019").pad("05-01-2020")
        # Method at test
        ts_trimmed = ts_pad.trim()
        # test if no NaNs
        self.assertFalse(ts_trimmed.data.isna().values.any())
        # test boundaries
        new_start, new_end = ts_trimmed.boundaries()
        self.assertEqual(start, new_start)
        self.assertEqual(end, new_end)

    def test__trim__only_start(self):
        # Prepare TimeSeries with int as values
        start = Timestamp("01-01-2020")
        end = Timestamp("03-01-2020")
        ts_initial = TimeSeries.create(start, end, "H")
        ts_initial = ts_initial.fill(42)
        # Pad the TimeSeries with np.nans
        pad_start = Timestamp("01-12-2019")
        pad_end = Timestamp("05-01-2020")
        ts_pad = ts_initial.pad(pad_start).pad(pad_end)
        # Method at test
        ts_trimmed = ts_pad.trim(side="start")
        # test if NaNs
        self.assertTrue(ts_trimmed.data.isna().values.any())
        # test boundaries
        new_start, new_end = ts_trimmed.boundaries()
        self.assertEqual(start, new_start)
        self.assertEqual(pad_end, new_end)

    def test__trim__only_end(self):
        # Prepare TimeSeries with int as values
        start = Timestamp("01-01-2020")
        end = Timestamp("03-01-2020")
        ts_initial = TimeSeries.create(start, end, "H")
        ts_initial = ts_initial.fill(42)
        # Pad the TimeSeries with np.nans
        pad_start = Timestamp("01-12-2019")
        pad_end = Timestamp("05-01-2020")
        ts_pad = ts_initial.pad(pad_start).pad(pad_end)
        # Method at test
        ts_trimmed = ts_pad.trim(side="end")
        # test if NaNs
        self.assertTrue(ts_trimmed.data.isna().values.any())
        # test boundaries
        new_start, new_end = ts_trimmed.boundaries()
        self.assertEqual(pad_start, new_start)
        self.assertEqual(end, new_end)

    def test__merge(self):
        # Prepare test
        ts1 = TimeSeries.create("01-2020", "03-2020", "H")
        ts2 = TimeSeries.create("02-2020", "04-2020", "H")
        # Call function
        ts = ts1.merge(ts2)
        # Test if index is monotonic increasing
        self.assertTrue(ts.data.index.is_monotonic_increasing)
        # Test if all values are there
        len1 = len(ts1) + len(ts2)
        len2 = len(ts)
        self.assertTrue(len1 == len2)

    def test__apply__on_self(self):
        # Prepare test
        val = 21
        ts = TimeSeries.create("01-2020", "02-2020", "H").fill(val)
        # Method at test
        ts = ts.apply(lambda x: x * 2)
        # Test
        for i in ts:
            self.assertEqual(i, val * 2)

    def test__apply__on_other_time_series(self):
        # Prepare test
        val_1 = 21
        val_2 = 3
        ts_1 = TimeSeries.create("01-2020", "02-2020", "H").fill(val_1)
        ts_2 = TimeSeries.create("01-2020", "02-2020", "H").fill(val_2)
        # Method at test
        ts = ts_1.apply(lambda x, y: x * y, ts_2)
        # Test
        for i in ts:
            self.assertEqual(i, val_1 * val_2)

    def test__apply__on_other_time_series_with_different_length(self):
        # Prepare test
        val_1 = 21
        val_2 = 3
        ts_1 = TimeSeries.create("01-2020", "02-2020", "H").fill(val_1)
        ts_2 = TimeSeries.create("01-2020", "04-2020", "H").fill(val_2)
        # Method at test
        with self.assertRaises(AssertionError):
            ts = ts_1.apply(lambda x, y: x * y, ts_2)

    def test__time_deltas(self):
        ts = TimeSeries.create("01-2020", "03-2020", "H")
        deltas = ts.time_detlas()
        for i in deltas[1:]:
            self.assertEqual(i, 3600.0)
            self.assertIs(type(i), float)

    def test__to_text__without_metadata(self):
        path = self.target_dir + "/to_text_without_metadata"
        my_time_series = TimeSeries(self.my_series)
        my_time_series.to_text(path)

        data_csv_path = "{}/{}.{}".format(path, TIME_SERIES_FILENAME, TIME_SERIES_EXT)
        does_data_csv_exist = os_path.exists(data_csv_path)
        self.assertTrue(does_data_csv_exist)

        meta_json_path = "{}/{}.{}".format(path, METADATA_FILENAME, METADATA_EXT)
        does_meta_json_exist = os_path.exists(meta_json_path)
        self.assertFalse(does_meta_json_exist)

    def test__to_text__with_metadata(self):
        path = self.target_dir + "/to_text_with_metadata"
        self.my_time_series.to_text(path)

        data_csv_path = "{}/{}.{}".format(path, TIME_SERIES_FILENAME, TIME_SERIES_EXT)
        does_data_csv_exist = os_path.exists(data_csv_path)
        self.assertTrue(does_data_csv_exist)

        meta_json_path = "{}/{}.{}".format(path, METADATA_FILENAME, METADATA_EXT)
        does_meta_json_exist = os_path.exists(meta_json_path)
        self.assertTrue(does_meta_json_exist)

    def test__to_pickle(self):
        pickle_path = "{}/{}.{}".format(self.target_dir, DEFAULT_EXPORT_FILENAME, PICKLE_EXT)
        self.my_time_series.to_pickle(pickle_path)
        does_pickle_exist = os_path.exists(pickle_path)
        self.assertTrue(does_pickle_exist)

    def test__to_df(self):
        df = self.my_time_series.to_df()
        self.assertTrue(df['values'].equals(self.my_series))
        self.assertIsInstance(df, DataFrame)

    def test__to_darts__type_check(self):
        ts = TimeSeries.create("01-2020", "02-2020", "H")
        ts = ts.fill(np.random.randint(0, 1000, len(ts)))
        self.assertIsInstance(ts, TimeSeries)
        dts = ts.to_darts()
        from darts import TimeSeries as DartsTimeSeries
        self.assertIsInstance(dts, DartsTimeSeries)

    def test__to_darts__series_equality(self):
        ts = TimeSeries.create("01-2020", "02-2020", "H")
        ts = ts.fill(np.random.randint(0, 1000, len(ts)))
        dts = ts.to_darts()
        is_equal = ts.data[TIME_SERIES_VALUES].equals(dts.pd_series())
        self.assertTrue(is_equal)

    def tearDown(self) -> None:
        del self.my_time_series
        shutil.rmtree(self.target_dir, ignore_errors=True)
