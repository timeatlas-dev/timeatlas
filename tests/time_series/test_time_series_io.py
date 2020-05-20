import shutil
from os import path as os_path
from unittest import TestCase
from pandas import DatetimeIndex, Series

from timeatlas import TimeSeries, Metadata
from timeatlas.config.constants import *


class TestTimeSeriesIO(TestCase):

    def setUp(self) -> None:

        # Create a time indexed series
        index = DatetimeIndex(['2019-01-01', '2019-01-02',
                               '2019-01-03', '2019-01-04'])
        self.my_series = Series([0.4, 1.0, 0.7, 0.6], index=index)

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

        self.my_time_series = TimeSeries(self.my_series, self.my_metadata)

        # Define a target directory
        self.target_dir = "../data/test-export"

    def test__TimeSeries_IO__to_text_without_metadata(self):
        path = self.target_dir + "/to_text_without_metadata"
        my_time_series = TimeSeries(self.my_series)
        my_time_series.to_text(path)

        data_csv_path = "{}/{}.{}".format(path, TIME_SERIES_FILENAME, TIME_SERIES_EXT)
        does_data_csv_exist = os_path.exists(data_csv_path)
        self.assertTrue(does_data_csv_exist)

        meta_json_path = "{}/{}.{}".format(path, METADATA_FILENAME, METADATA_EXT)
        does_meta_json_exist = os_path.exists(meta_json_path)
        self.assertFalse(does_meta_json_exist)

    def test__TimeSeries_IO__to_text_with_metadata(self):
        path = self.target_dir + "/to_text_with_metadata"
        self.my_time_series.to_text(path)

        data_csv_path = "{}/{}.{}".format(path, TIME_SERIES_FILENAME, TIME_SERIES_EXT)
        does_data_csv_exist = os_path.exists(data_csv_path)
        self.assertTrue(does_data_csv_exist)

        meta_json_path = "{}/{}.{}".format(path, METADATA_FILENAME, METADATA_EXT)
        does_meta_json_exist = os_path.exists(meta_json_path)
        self.assertTrue(does_meta_json_exist)

    def test__TimeSeries_IO__to_pickle(self):
        pickle_path = "{}/{}.{}".format(self.target_dir, DEFAULT_EXPORT_FILENAME, PICKLE_EXT)
        self.my_time_series.to_pickle(pickle_path)
        does_pickle_exist = os_path.exists(pickle_path)
        self.assertTrue(does_pickle_exist)

    def tearDown(self) -> None:
        del self.my_time_series
        shutil.rmtree(self.target_dir, ignore_errors=True)

