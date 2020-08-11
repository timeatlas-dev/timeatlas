from unittest import TestCase
import os
import shutil

import timeatlas as ta
from timeatlas import TimeSeriesDataset
from timeatlas.config.constants import *


class TestTimeSeriesDatasetIO(TestCase):

    def setUp(self) -> None:
        self.target_dir = "data/test-import/to_text_with_metadata"
        ts_1 = ta.read_text(self.target_dir)
        ts_2 = ta.read_text(self.target_dir)
        self.tsd = TimeSeriesDataset([ts_1, ts_2])
        self.target_dir = "data/test-export"

    def test__TimeSeriesDataset_IO__to_text(self):
        my_path = "{}/{}".format(self.target_dir, "tsd_to_text")
        self.tsd.to_text(my_path)
        number_of_dirs = len(next(os.walk(my_path))[1])
        # Checking the number of dirs in my_path is enough to
        # ensure export dir existence and presence of TimeSeries
        # export in text files
        self.assertTrue(number_of_dirs == 2)

    def test__TimeSeriesDataset_IO__to_pickle(self):
        pickle_path = "{}/{}.{}".format(self.target_dir, DEFAULT_EXPORT_FILENAME, PICKLE_EXT)
        self.tsd.to_pickle(pickle_path)
        does_pickle_exist = os.path.exists(pickle_path)
        self.assertTrue(does_pickle_exist)

    def tearDown(self) -> None:
        del self.tsd
        shutil.rmtree(self.target_dir, ignore_errors=True)
