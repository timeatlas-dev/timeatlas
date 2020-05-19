import shutil
from unittest import TestCase
from pandas import DatetimeIndex, Series

from timeatlas import TimeSeries, Metadata


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

    def test__TimeSeries_IO__to_text_with_metadata(self):
        path = self.target_dir + "/to_text_with_metadata"
        self.my_time_series.to_text(path)

    def test__TimeSeries_IO__to_pickle(self):
        path = self.target_dir + "/to_pickle.pickle"
        self.my_time_series.to_pickle(path)

    def tearDown(self) -> None:
        del self.my_time_series
        shutil.rmtree(self.target_dir, ignore_errors=True)

