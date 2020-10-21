from unittest import TestCase
from pandas import DataFrame
import timeatlas as ta
from timeatlas import TimeSeries, Metadata, TimeSeriesDataset


class TestReadWrite(TestCase):

    def setUp(self) -> None:
        self.target_dir = "./test/data/test-import"

    def test__ReadWrite__read_text_without_metadata(self):
        wo = "{}/{}".format(self.target_dir, "to_text_without_metadata")
        ts = ta.read_text(wo)
        self.assertIsNone(ts.metadata)
        self.assertIsInstance(ts.series, DataFrame)
        self.assertIsInstance(ts, TimeSeries)

    def test__ReadWrite__read_text_with_metadata(self):
        w = "{}/{}".format(self.target_dir, "to_text_with_metadata")
        ts = ta.read_text(w)
        self.assertIsInstance(ts.metadata, Metadata)
        self.assertIsInstance(ts.series, DataFrame)
        self.assertIsInstance(ts, TimeSeries)

    def test__ReadWrite__read_tsd_without_metadata(self):
        wo = "{}/{}".format(self.target_dir, "read_tsd_without_metadata")

        # setup data
        ts_wo = TimeSeries.create("01-01-1990", "01-03-1990", "1D")
        ts_wo.series["values"] = [0, 1, 2]
        tsd_wo = TimeSeriesDataset([ts_wo, ts_wo, ts_wo])
        tsd_wo.to_text(wo)

        # load data
        tsd = ta.read_tsd(wo)

        # test that all TS both TSDs are equal
        check = True
        for i, ts in enumerate(tsd):
            check &= ts.series.equals(tsd_wo[i].series)

        # assertions
        self.assertTrue(check)
        self.assertIsInstance(tsd, TimeSeriesDataset)

    def test__ReadWrite__read_tsd_with_metadata(self):
        w = "{}/{}".format(self.target_dir, "read_tsd_with_metadata")

        # setup data
        ts_w = TimeSeries.create("01-01-1990", "01-03-1990", "1D")
        ts_w.series["values"] = [0, 1, 2]
        ts_w.series["label_test"] = [0, None, 2]
        tsd_w = TimeSeriesDataset([ts_w, ts_w, ts_w])
        tsd_w.to_text(w)

        # load data
        tsd = ta.read_tsd(w)

        # test that all TS both TSDs are equal
        check = True
        for i, ts in enumerate(tsd):
            check &= ts.series.equals(tsd_w[i].series)
            check &= (ts.label == tsd_w[i].label)

        # assertions
        self.assertTrue(check)
        self.assertIsInstance(tsd, TimeSeriesDataset)

    def test__ReadWrite__csv_to_tsd(self):
        w = "{}/{}".format(self.target_dir, "csv-to-tsd")
        # setup data
        ts_w = TimeSeries.create("01-01-1990", "01-03-1990", "1D")
        ts_w.series["values"] = [0, 1, 2]
        ts_w.series["label_test"] = [0, None, 2]
        tsd_w = TimeSeriesDataset([ts_w, ts_w, ts_w])
        df_tsd = tsd_w.to_df()
        # save tsd as csv
        df_tsd.to_csv(f'{w}/tsd.csv')

        # load data
        tsd = ta.csv_to_tsd(f'{w}/tsd.csv')
        self.assertIsInstance(tsd, TimeSeriesDataset)

        check = True
        for i, ts in enumerate(tsd):
            check &= ts.series.equals(tsd_w[i].series)
            check &= (ts.label == tsd_w[i].label)

        self.assertTrue(check)
