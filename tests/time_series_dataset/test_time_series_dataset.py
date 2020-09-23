from unittest import TestCase

from pandas import Timestamp
import numpy as np

from timeatlas import TimeSeries, TimeSeriesDataset
import timeatlas as ta


class TestTimeSeriesDataset(TestCase):

    def test__TimeSeriesDataset__is_instance(self):
        my_time_series_dataset = TimeSeriesDataset()
        self.assertIsInstance(my_time_series_dataset, TimeSeriesDataset)

    def test__TimeSeriesDataset__construct(self):
        ts1 = TimeSeries.create("01-01-2020", "01-02-2020", "H").fill(0)
        ts2 = TimeSeries.create("01-01-2020", "01-03-2020", "H").fill(0)
        my_time_series_dataset = TimeSeriesDataset([ts1, ts2])
        self.assertTrue(my_time_series_dataset.len() == 2)
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
        for ts in tsd.data:
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
        for ts in tsd.data:
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
            for ts in tsd.data:
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

    def test__TimeSeriesDataset__add_component(self):
        # Create series
        tsd = TimeSeriesDataset()
        ts_1 = TimeSeries.create("01-2020", "02-2020", "H")
        ts_2 = TimeSeries.create("01-2020", "03-2020", "H")
        # Test
        self.assertTrue(tsd.len() == 0)
        tsd.add_component(ts_1)
        self.assertTrue(tsd.len() == 1)
        tsd.add_component(ts_2)
        self.assertTrue(tsd.len() == 2)

    def test__TimeSeriesDataset__remove_component(self):
        # Create series
        ts_1 = TimeSeries.create("01-2020", "02-2020", "H")
        ts_2 = TimeSeries.create("01-2020", "03-2020", "H")
        my_arr = [ts_1, ts_2]
        tsd = TimeSeriesDataset(my_arr)
        # Test
        self.assertTrue(tsd.len() == 2)
        tsd.remove_component(-1)
        self.assertTrue(tsd.len() == 1)
        tsd.remove_component(-1)
        self.assertTrue(tsd.len() == 0)
