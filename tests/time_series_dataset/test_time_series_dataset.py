from unittest import TestCase
from glob import glob
import os
import shutil

from pandas import Timestamp, DataFrame
from pandas.tseries.frequencies import to_offset
import numpy as np

from timeatlas import TimeSeries, TimeSeriesDataset
from timeatlas.config.constants import TIME_SERIES_VALUES


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

    def test__TimeSeriesDataset__copy__shallow(self):
        # params
        ts_1 = TimeSeries.create("01-2020", "03-2020", "H")
        ts_2 = TimeSeries.create("02-2020", "04-2020", "min")
        my_arr = [ts_1, ts_2]
        # object creation
        tsd = TimeSeriesDataset(my_arr)
        # test
        copy = tsd.copy(deep=False)
        self.assertNotEqual(id(tsd), id(copy))
        for i in range(len(copy)):
            self.assertEqual(id(tsd[i]), id(copy[i]))

    def test__TimeSeriesDataset__copy__deep(self):
        # params
        ts_1 = TimeSeries.create("01-2020", "03-2020", "H")
        ts_2 = TimeSeries.create("02-2020", "04-2020", "min")
        my_arr = [ts_1, ts_2]
        # object creation
        tsd = TimeSeriesDataset(my_arr)
        # test
        copy = tsd.copy()  # defaults to deep copy
        self.assertNotEqual(id(tsd), id(copy))
        for i in range(len(copy)):
            self.assertNotEqual(id(tsd[i]), id(copy[i]))

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
            for i in ts.series[TIME_SERIES_VALUES]:
                self.assertIs(i, np.nan)
        # fill with zeros
        tsd = tsd.fill(0)
        # test if every elements are zeros
        for ts in tsd:
            for i in ts.series[TIME_SERIES_VALUES]:
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
            for i in ts.series[TIME_SERIES_VALUES]:
                self.assertIs(i, 0)
        # fill with Nans
        tsd = tsd.empty()
        # test if every elements are nans
        for ts in tsd:
            for i in ts.series[TIME_SERIES_VALUES]:
                self.assertTrue(np.isnan(i))

    def test__TimeSeriesDataset__pad_right(self):
        goal_left = "2020-01-01 00:00:00"

        # Create series
        ts1 = TimeSeries.create("01-02-2020", "01-08-2020", "1D")
        ts2 = TimeSeries.create("01-04-2020", "01-09-2020", "1D")

        tsd = TimeSeriesDataset([ts1, ts2])

        tsd_padded = tsd.pad(limit=goal_left)

        # check if all have the same left boundary
        [print(str(ts.boundaries()[0])) for ts in tsd_padded]
        [print(goal_left) for ts in tsd_padded]

        self.assertTrue(all([str(ts.boundaries()[0]) == goal_left for ts in tsd_padded]))

    def test__TimeSeriesDataset__pad_left(self):
        goal_right = "2020-01-10 00:00:00"

        # Create series
        ts1 = TimeSeries.create("01-02-2020", "01-08-2020", "1D")
        ts2 = TimeSeries.create("01-04-2020", "01-09-2020", "1D")

        tsd = TimeSeriesDataset([ts1, ts2])

        tsd_padded = tsd.pad(limit=goal_right)

        # check if all have the same left boundary
        self.assertTrue(all([str(ts.boundaries()[1]) == goal_right for ts in tsd_padded]))

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
            for i in ts.series[TIME_SERIES_VALUES]:
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

    def test__TimeSeriesDataset__merge_by_label(self):
        # Create TSD
        ts_1 = TimeSeries.create('2019-01-01', '2019-01-02', "1D")
        ts_1.series[TIME_SERIES_VALUES] = [0, 1]
        ts_1.label = "Sensor1"

        ts_2 = TimeSeries.create('2019-01-03', '2019-01-03', "1D")
        ts_2.series[TIME_SERIES_VALUES] = [2]
        ts_2.label = "Sensor1"

        tsd1 = TimeSeriesDataset([ts_1, ts_2])

        ts_3 = TimeSeries.create('2019-01-01', '2019-01-03', "1D")
        ts_3.series[TIME_SERIES_VALUES] = [0, 1, 2]
        ts_3.label = "Sensor2"

        tsd2 = TimeSeriesDataset([ts_3])

        tsd_merged = tsd1.merge_by_label(tsd2)
        self.assertIsInstance(tsd_merged, TimeSeriesDataset)

        # Create Goal
        ts_goal_1 = TimeSeries.create('2019-01-01', '2019-01-03', "1D")
        ts_goal_1.series[TIME_SERIES_VALUES] = [0, 1, 2]
        ts_goal_1.label = "Sensor1"

        ts_goal_2 = TimeSeries.create('2019-01-01', '2019-01-03', "1D")
        ts_goal_2.series[TIME_SERIES_VALUES] = [0, 1, 2]
        ts_goal_2.label = "Sensor2"

        tsd_goal = TimeSeriesDataset([ts_goal_1, ts_goal_2])

        check = True
        for i, ts in enumerate(tsd_goal):
            check &= ts.series.equals(tsd_merged[i].series)
            check &= (ts.label == tsd_merged[i].label)

        self.assertTrue(check)

    def test__TimeSeriesDataset__shuffle(self):
        # Create TSD
        ts_1 = TimeSeries.create('2019-01-03', '2019-01-03', "1D")
        ts_1.series[TIME_SERIES_VALUES] = [2]
        ts_1.label = "Sensor1"
        ts_2 = TimeSeries.create('2019-01-03', '2019-01-03', "1D")
        ts_2.series[TIME_SERIES_VALUES] = [2]
        ts_2.label = "Sensor2"
        ts_3 = TimeSeries.create('2019-01-03', '2019-01-03', "1D")
        ts_3.series[TIME_SERIES_VALUES] = [2]
        ts_3.label = "Sensor3"
        ts_4 = TimeSeries.create('2019-01-03', '2019-01-03', "1D")
        ts_4.series[TIME_SERIES_VALUES] = [2]
        ts_4.label = "Sensor4"

        tsd = TimeSeriesDataset([ts_1, ts_2, ts_3, ts_4])

        tsd_shuffled = tsd.shuffle(inplace=False)

        # check if the label is the same -> can fail with a low probability
        check = True
        for i, ts in enumerate(tsd):
            check &= (ts.label == tsd_shuffled[i].label)

        self.assertFalse(check)

    def test__TimeSeriesDataset__apply(self):
        # TODO: Not implemented
        pass

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

    def test__TimeSeriesDataset__regularize_intersect(self):
        # The goal of the right boundary
        goal_left = "2019-01-05 00:00:00"
        goal_right = "2019-01-05 00:00:00"

        # setup data
        ts_1 = TimeSeries.create('2019-01-03', '2019-01-05', "1D")
        ts_1.series[TIME_SERIES_VALUES] = [3, 4, 5]
        ts_2 = TimeSeries.create('2019-01-01', '2019-01-05', "1D")
        ts_2.series[TIME_SERIES_VALUES] = [1, 2, 3, 4, 5]
        ts_3 = TimeSeries.create('2019-01-05', '2019-01-10', "1D")
        ts_3.series[TIME_SERIES_VALUES] = [5, 6, 7, 8, 9, 10]
        ts_4 = TimeSeries.create('2019-01-03', '2019-01-07', "1D")
        ts_4.series[TIME_SERIES_VALUES] = [3, 4, 5, 6, 7]

        tsd = TimeSeriesDataset([ts_1, ts_2, ts_3, ts_4])

        # regularize to the right
        tsd = tsd.regularize(side="][")

        # assert that all TS in TSD end in right_goal
        self.assertTrue(all([str(ts.boundaries()[0]) == goal_left for ts in tsd]))
        self.assertTrue(all([str(ts.boundaries()[1]) == goal_right for ts in tsd]))

    def test__TimeSeriesDataset__regularize_left(self):
        # The goal of the right boundary
        goal_right = "2019-01-01 00:00:00"

        # setup data
        ts_1 = TimeSeries.create('2019-01-03', '2019-01-05', "1D")
        ts_1.series[TIME_SERIES_VALUES] = [3, 4, 5]
        ts_2 = TimeSeries.create('2019-01-01', '2019-01-05', "1D")
        ts_2.series[TIME_SERIES_VALUES] = [1, 2, 3, 4, 5]
        ts_3 = TimeSeries.create('2019-01-05', '2019-01-10', "1D")
        ts_3.series[TIME_SERIES_VALUES] = [5, 6, 7, 8, 9, 10]
        ts_4 = TimeSeries.create('2019-01-03', '2019-01-07', "1D")
        ts_4.series[TIME_SERIES_VALUES] = [3, 4, 5, 6, 7]

        tsd = TimeSeriesDataset([ts_1, ts_2, ts_3, ts_4])

        # regularize to the right
        tsd = tsd.regularize(side="[[")

        # assert that all TS in TSD end in right_goal
        self.assertTrue(all([str(ts.boundaries()[0]) == goal_right for ts in tsd]))

    def test__TimeSeriesDataset__regularize_right(self):
        # The goal of the right boundary
        goal_right = "2019-01-10 00:00:00"

        # setup data
        ts_1 = TimeSeries.create('2019-01-03', '2019-01-05', "1D")
        ts_1.series[TIME_SERIES_VALUES] = [3, 4, 5]
        ts_2 = TimeSeries.create('2019-01-01', '2019-01-05', "1D")
        ts_2.series[TIME_SERIES_VALUES] = [1, 2, 3, 4, 5]
        ts_3 = TimeSeries.create('2019-01-05', '2019-01-10', "1D")
        ts_3.series[TIME_SERIES_VALUES] = [5, 6, 7, 8, 9, 10]
        ts_4 = TimeSeries.create('2019-01-03', '2019-01-07', "1D")
        ts_4.series[TIME_SERIES_VALUES] = [3, 4, 5, 6, 7]

        tsd = TimeSeriesDataset([ts_1, ts_2, ts_3, ts_4])

        # regularize to the right
        tsd = tsd.regularize(side="]]")

        # assert that all TS in TSD end in right_goal
        self.assertTrue(all([str(ts.boundaries()[1]) == goal_right for ts in tsd]))

    def test__TimeSeriesDataset__regularize_union(self):
        # The goal of the right boundary
        goal_left = "2019-01-01 00:00:00"
        goal_right = "2019-01-10 00:00:00"

        # setup data
        ts_1 = TimeSeries.create('2019-01-03', '2019-01-05', "1D")
        ts_1.series[TIME_SERIES_VALUES] = [3, 4, 5]
        ts_2 = TimeSeries.create('2019-01-01', '2019-01-05', "1D")
        ts_2.series[TIME_SERIES_VALUES] = [1, 2, 3, 4, 5]
        ts_3 = TimeSeries.create('2019-01-05', '2019-01-10', "1D")
        ts_3.series[TIME_SERIES_VALUES] = [5, 6, 7, 8, 9, 10]
        ts_4 = TimeSeries.create('2019-01-03', '2019-01-07', "1D")
        ts_4.series[TIME_SERIES_VALUES] = [3, 4, 5, 6, 7]

        tsd = TimeSeriesDataset([ts_1, ts_2, ts_3, ts_4])

        # regularize to the right
        tsd = tsd.regularize(side="[]")

        # assert that all TS in TSD end in right_goal
        self.assertTrue(all([str(ts.boundaries()[0]) == goal_left for ts in tsd]))
        self.assertTrue(all([str(ts.boundaries()[1]) == goal_right for ts in tsd]))

    def test__TimeSeriesDataset__to_text(self):
        out_dir = '../data/test-import/tsd_to_text/'

        ts = TimeSeries.create("01-01-1990", "01-03-1990", "1D")

        # prepare data

        ts.series[TIME_SERIES_VALUES] = [0, 1, 2]
        ts.series["label_test"] = [0, None, 2]
        ts.label = "Test"

        tsd = TimeSeriesDataset([ts, ts, ts])
        tsd.to_text(out_dir)

        # preparte test variables
        goal_length = len(tsd)

        # check that it created the three folder
        folders = glob(f"{out_dir}/[0-9]")
        self.assertEqual(len(folders), goal_length)

        # check that all folders have a data.csv and a metadata.json
        check = True
        for dir in folders:
            check &= os.path.isfile(f'{dir}/data.csv')
            check &= os.path.isfile(f'{dir}/meta.json')
        self.assertTrue(check)

        # clean up
        shutil.rmtree(out_dir)
        # check if cleaned
        self.assertFalse(os.path.isdir(out_dir))

    def test__TimeSeriesDataset__to_pickle(self):
        out_dir = '../data/test-import/tsd_to_pickle/'

        ts = TimeSeries.create("01-01-1990", "01-03-1990", "1D")

        # prepare data

        ts.series[TIME_SERIES_VALUES] = [0, 1, 2]
        ts.series["label_test"] = [0, None, 2]
        ts.label = "Test"

        tsd = TimeSeriesDataset([ts, ts, ts])
        tsd.to_pickle(f"{out_dir}/tsd.pkl")

        self.assertTrue(os.path.isfile(f"{out_dir}/tsd.pkl"))

        # clean up
        shutil.rmtree(out_dir)
        # check if cleaned
        self.assertFalse(os.path.isdir(out_dir))

    def test__TimeSeriesDataset__to_df(self):
        # goal_df
        df_goal = DataFrame([[0, 0, 0, 0], [1, None, 1, None]],
                            columns=['0_values', '0_label_test', '1_values', '1_label_test'],
                            index=["01-01-1990", "01-02-1990"])

        ts = TimeSeries.create("01-01-1990", "01-02-1990", "1D")

        # prepare data

        ts.series[TIME_SERIES_VALUES] = [0, 1]
        ts.series["label_test"] = [0, None]
        ts.label = "Test"
        tsd = TimeSeriesDataset([ts, ts])

        df = tsd.to_df()

        self.assertTrue(df.equals(df_goal))
