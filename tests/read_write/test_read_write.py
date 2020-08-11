from unittest import TestCase
from pandas import Series
import timeatlas as ta
from timeatlas import TimeSeries, Metadata


class TestReadWrite(TestCase):

    def setUp(self) -> None:
        self.target_dir = "../data/test-import"

    def test__ReadWrite__read_text_without_metadata(self):
        wo = "{}/{}".format(self.target_dir, "to_text_without_metadata")
        ts = ta.read_text(wo)
        self.assertIsNone(ts.metadata)
        self.assertIsInstance(ts.series, Series)
        self.assertIsInstance(ts, TimeSeries)

    def test__ReadWrite__read_text_with_metadata(self):
        w = "{}/{}".format(self.target_dir, "to_text_with_metadata")
        ts = ta.read_text(w)
        self.assertIsInstance(ts.metadata, Metadata)
        self.assertIsInstance(ts.series, Series)
        self.assertIsInstance(ts, TimeSeries)
