from unittest import TestCase

from pandas import Timestamp

from timeatlas import TimeSeries, TimeSeriesDataset
import timeatlas as ta


class TestTimeSeriesDataset(TestCase):

    def test__TimeSeriesDataset__is_instance(self):
        my_time_series_dataset = TimeSeriesDataset()
        self.assertIsInstance(my_time_series_dataset, TimeSeriesDataset)

    def test__TimeSeriesDataset__construct(self):
        my_arr = [self.my_time_series_1, self.my_time_series_2]
        my_time_series_dataset = TimeSeriesDataset(my_arr)
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
        # test
        self.assertEqual(len(chunkified_tds), n_chunks)

    def test__TimeSeriesDataset__fill(self):
        pass

    def test__TimeSeriesDataset__empty(self):
        pass

    def test__TimeSeriesDataset__trim(self):
        pass

    def test__TimeSeriesDataset__merge(self):
        pass

    def test__TimeSeriesDataset__add(self):
        tsd = TimeSeriesDataset()
        self.assertTrue(tsd.len() == 0)
        tsd.add(self.my_time_series_1)
        self.assertTrue(tsd.len() == 1)
        tsd.add(self.my_time_series_2)
        self.assertTrue(tsd.len() == 2)

    def test__TimeSeriesDataset__remove(self):
        my_arr = [self.my_time_series_1, self.my_time_series_2]
        tsd = TimeSeriesDataset(my_arr)
        self.assertTrue(tsd.len() == 2)
        tsd.remove(-1)
        self.assertTrue(tsd.len() == 1)
        tsd.remove(-1)
        self.assertTrue(tsd.len() == 0)

