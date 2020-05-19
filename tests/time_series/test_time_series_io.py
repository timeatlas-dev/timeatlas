from unittest import TestCase
from pandas import DatetimeIndex, Series

from timeatlas import TimeSeries, MetadataIO


class TestTimeSeriesIO(TestCase):

    def setUp(self) -> None:

        # Create a TimeSeries Object
        index = DatetimeIndex(['2019-01-01', '2019-01-02',
                               '2019-01-03', '2019-01-04'])
        my_series = Series([0.4, 1.0, 0.7, 0.6], index=index)
        my_metadata = MetadataIO()
        self.my_time_series = TimeSeries(my_series, my_metadata)

        # Define a target directory
        self.target_dir = "../data/test-export"

    def test__TimeSeriesIO__write(self):
        self.my_time_series.write(self.target_dir, "my")

    def tearDown(self) -> None:
        del self.my_time_series
