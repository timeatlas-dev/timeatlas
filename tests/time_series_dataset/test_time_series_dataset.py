from unittest import TestCase

from pandas import Timestamp
from pandas.tseries.frequencies import to_offset
import numpy as np

from timeatlas import TimeSeries, TimeSeriesDataset


class TestTimeSeriesDataset(TestCase):

    def test__TimeSeriesDataset__is_instance(self):
        my_time_series_dataset = TimeSeriesDataset()
        self.assertIsInstance(my_time_series_dataset, TimeSeriesDataset)

    def test__TimeSeriesDataset__construct(self):
        ts1 = TimeSeries.create("01-01-2020", "01-02-2020", "H").fill(0)
        ts2 = TimeSeries.create("01-01-2020", "01-03-2020", "H").fill(0)
        my_time_series_dataset = TimeSeriesDataset([ts1, ts2])
        self.assertTrue(len(my_time_series_dataset) == 2)
        self.assertIsInstance(my_time_series_dataset, TimeSeriesDataset)

    def test__TimeSeriesDataset__create(self):
        # params
        length = 2
        start = "01-01-2020"
        end = "02-01-2020"
        # object creation
        tsd = TimeSeriesDataset.create(length, start, end)
        # test
        self.assertIsInstance(tsd, TimeSeriesDataset)
        # Test if length is the same as planned
        self.assertEqual(len(tsd), length)

    def test__TimeSeriesDataset__create_with_freq_as_str(self):
        # params
        length = 3
        start = "01-01-2020"
        end = "02-01-2020"
        freq = "H"
        # object creation
        tsd = TimeSeriesDataset.create(length, start, end, freq)
        # test
        for ts in tsd:
            # if equal, all ts in tsd have an hourly frequency
            self.assertEqual(ts.frequency(), "H")

    def test__TimeSeriesDataset__create_with_freq_as_time_series(self):
        # params
        length = 3
        start = "01-01-2020"
        end = "02-01-2020"
        freq = TimeSeries.create("01-01-2020", "01-02-2020", "H")
        # object creation
        tsd = TimeSeriesDataset.create(length, start, end, freq)
        # test
        for ts in tsd:
            # if equal, all ts in tsd have an hourly frequency
            self.assertEqual(ts.frequency(), freq.frequency())

    def test__TimeSeriesDataset__split_at(self):
        # params
        length = 3
        start = "01-01-2020 00:00"
        end = "03-01-2020 00:00"
        freq = "H"
        # object creation
        tsd = TimeSeriesDataset.create(length, start, end, freq)
        # test
        splitting_point = "02-01-2020 00:00"
        tsd1, tsd2 = tsd.split_at(splitting_point)

        # check the first split
        for ts in tsd1:
            self.assertEqual(ts.start(), Timestamp(start))
            self.assertEqual(ts.end(), Timestamp(splitting_point))

        # check the second split
        for ts in tsd2:
            self.assertEqual(ts.start(), Timestamp(splitting_point))
            self.assertEqual(ts.end(), Timestamp(end))

    def test__TimeSeriesDataset__split_in_chunks(self):
        # params
        n_chunks = 12
        length = 3
        start = "01-01-2020"
        end = "02-01-2020"
        freq = "H"
        # object creation
        tsd = TimeSeriesDataset.create(length, start, end, freq)
        chunkified_tds = tsd.split_in_chunks(n_chunks)
        # test if the number of chunks is the right one
        self.assertEqual(len(chunkified_tds), n_chunks)
        for tsd in chunkified_tds:
            # test if each item is a TimeSeriesDataset
            self.assertIsInstance(tsd, TimeSeriesDataset)
            # test if each item in a TimeSeriesDataset is a TS
            for ts in tsd:
                self.assertIsInstance(ts, TimeSeries)

    def test__TimeSeriesDataset__fill(self):
        # params
        length = 3
        start = "01-01-2020"
        end = "02-01-2020"
        freq = "H"
        # object creation
        tsd = TimeSeriesDataset.create(length, start, end, freq)
        # test if every elements are nans
        for ts in tsd:
            for i in ts.series["values"]:
                self.assertIs(i, np.nan)
        # fill with zeros
        tsd = tsd.fill(0)
        # test if every elements are zeros
        for ts in tsd:
            for i in ts.series["values"]:
                self.assertIs(i, 0)

    def test__TimeSeriesDataset__empty(self):
        # params
        length = 3
        start = "01-01-2020"
        end = "02-01-2020"
        freq = "H"
        # object creation
        tsd = TimeSeriesDataset.create(length, start, end, freq)
        tsd = tsd.fill(0)
        # test if every elements are zeros
        for ts in tsd:
            for i in ts.series["values"]:
                self.assertIs(i, 0)
        # fill with Nans
        tsd = tsd.empty()
        # test if every elements are nans
        for ts in tsd:
            for i in ts.series["values"]:
                self.assertTrue(np.isnan(i))

    def test__TimeSeriesDataset__trim(self):
        # Create series
        ts1 = TimeSeries.create("02-01-2020", "06-01-2020", "H").fill(0)
        ts2 = TimeSeries.create("01-01-2020", "04-01-2020", "H").fill(0)
        # Add Nones
        ts1.series[:21] = None
        ts1.series[-4:] = None
        ts2.series[:2] = None
        ts2.series[-14:] = None
        # Make the TSD
        tsd = TimeSeriesDataset([ts1, ts2])
        # Call the function to test
        tsd = tsd.trim()
        # Test
        for ts in tsd:
            for i in ts.series["values"]:
                self.assertFalse(np.isnan(i))

    def test__TimeSeriesDataset__merge(self):
        # Create the time series
        ts1 = TimeSeries.create("01-01-2020", "01-02-2020", "H").fill(0)
        ts2 = TimeSeries.create("01-01-2020", "01-03-2020", "H").fill(0)
        ts3 = TimeSeries.create("01-01-2020", "01-04-2020", "H").fill(0)
        ts4 = TimeSeries.create("01-02-2020", "01-05-2020", "H").fill(0)
        tsd1 = TimeSeriesDataset([ts1, ts2])
        tsd2 = TimeSeriesDataset([ts3, ts4])
        # Call the function
        tsd = tsd1.merge(tsd2)
        # Test the beginnings and the ends
        self.assertTrue(ts1.start() == tsd[0].start())
        self.assertTrue(ts3.end() == tsd[0].end())
        self.assertTrue(ts2.start() == tsd[1].start())
        self.assertTrue(ts4.end() == tsd[1].end())

    def test__TimeSeriesDataset__append(self):
        # Create series
        tsd = TimeSeriesDataset()
        ts_1 = TimeSeries.create("01-2020", "02-2020", "H")
        ts_2 = TimeSeries.create("01-2020", "03-2020", "H")
        # Test
        self.assertTrue(len(tsd) == 0)
        tsd.append(ts_1)
        self.assertTrue(len(tsd) == 1)
        tsd.append(ts_2)
        self.assertTrue(len(tsd) == 2)

    def test__TimeSeriesDataset__del(self):
        # Create series
        ts_1 = TimeSeries.create("01-2020", "02-2020", "H")
        ts_2 = TimeSeries.create("01-2020", "03-2020", "H")
        my_arr = [ts_1, ts_2]
        tsd = TimeSeriesDataset(my_arr)
        # Test
        self.assertTrue(len(tsd) == 2)
        del tsd[-1]
        self.assertTrue(len(tsd) == 1)
        del tsd[-1]
        self.assertTrue(len(tsd) == 0)

    def test__TimeSeriesDataset__resample(self):
        # Create series
        ts_1 = TimeSeries.create("01-2020", "02-2020", "H")
        ts_2 = TimeSeries.create("01-2020", "03-2020", "min")
        my_arr = [ts_1, ts_2]
        tsd = TimeSeriesDataset(my_arr)

        # Test lowest
        tsd_freq_before = tsd.frequency()
        lowest_freq = max([to_offset(f) for f in tsd_freq_before])
        tsd_res = tsd.resample(freq="lowest")
        for ts in tsd_res:
            current_offset = to_offset(ts.frequency())
            self.assertEqual(current_offset, lowest_freq)

        # Test highest
        tsd_freq_before = tsd.frequency()
        highest_freq = min([to_offset(f) for f in tsd_freq_before])
        tsd_res = tsd.resample(freq="highest")
        for ts in tsd_res:
            current_offset = to_offset(ts.frequency())
            self.assertEqual(current_offset, highest_freq)

        # Test Dateoffset str
        offest_str = "15min"
        tsd_res = tsd.resample(freq=offest_str)
        for ts in tsd_res:
            current_offset = to_offset(ts.frequency())
            self.assertEqual(current_offset, offest_str)

        # Test TimeSeries as arg
        offset_str_arg = "30min"
        ts_arg = TimeSeries.create("01-2020", "03-2020", offset_str_arg)
        tsd_res = tsd.resample(freq=ts_arg)
        for ts in tsd_res:
            current_offset = to_offset(ts.frequency())
            self.assertEqual(current_offset, offset_str_arg)

    def test__TimeSeriesDataset__merge_by_label(self):
        # Create TSD
        ts_1 = TimeSeries.create('2019-01-01', '2019-01-02', "1D")
        ts_1.label = "Sensor1"

        ts_2 = TimeSeries.create('2019-01-02', '2019-01-03', "1D")
        ts_2.label = "Sensor1"

        tsd1 = TimeSeriesDataset([ts_1, ts_2])

        ts_3 = TimeSeries.create('2019-01-02', '2019-01-03', "1D")
        ts_3.label = "Sensor2"

        tsd2 = TimeSeriesDataset([ts_3])

        tsd_merged = tsd1.merge_by_label(tsd2)

        # Create Goal
        ts_goal_1 = TimeSeries.create('2019-01-01', '2019-01-03', "1D")
        ts_1.label = "Sensor1"

        ts_goal_2 = TimeSeries.create('2019-01-02', '2019-01-03', "1D")
        ts_goal_2.label = "Sensor2"

        tsd_goal = TimeSeriesDataset([ts_goal_1, ts_goal_2])

        print(tsd_goal.boundaries())
        print(tsd_merged.boundaries())

        self.assertEqual(tsd_merged, tsd_goal)
        self.assertIsInstance(tsd_merged, TimeSeriesDataset)
