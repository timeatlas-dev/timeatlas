# standard libraries
from unittest import TestCase
# external modules
import pandas as pd
# internal modules
from timeatlas import TimeSeriesDarts


class TestTimeShop(TestCase):

    def setUp(self) -> None:
        df = pd.DataFrame(data={'First': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]})
        inds = [f'2021-01-{day}' for day in range(1, 11)]
        df.index = pd.to_datetime(inds)
        self.ts = TimeSeriesDarts.from_dataframe(df)

    def test__TimeShop__add_with_end_time(self):
        """

        Integration test for workflow select -> flatten -> add

        """
        # create target values
        df = pd.DataFrame(data={'First': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]})
        inds = [f'2021-01-{day}' for day in range(1, 11)]
        df.index = pd.to_datetime(inds)
        target = TimeSeriesDarts.from_dataframe(df)

        # create TimeShop object
        tss = self.ts.edit()
        # copy into the clipboard
        tss.select(other=tss.time_series, start_time=self.ts.start_time(), end_time=self.ts.end_time())
        # create values to add (this assumes that TimeShop.flatten() works)
        tss.flatten(value=1)
        # add the target to the object to be tested
        tss.add()

        # test that all values are the same
        self.assertTrue((tss.extract().values() == target.values()).all())

    def test__TimeShop__multiply_with_end_time(self):
        """

        Integration test for workflow select -> flatten -> multiply

        """
        # create target values
        df = pd.DataFrame(data={'First': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]})
        inds = [f'2021-01-{day}' for day in range(1, 11)]
        df.index = pd.to_datetime(inds)
        target = TimeSeriesDarts.from_dataframe(df)

        # create TimeShop object
        tss = self.ts.edit()
        # select into the clipboard
        tss.select(other=tss.time_series, start_time=self.ts.start_time(), end_time=self.ts.end_time())
        # create values to add (this assumes that TimeShop.flatten() works)
        tss.flatten(value=2)
        # multiply the target to the object to be tested
        tss.multiply()

        # test that all values are the same
        self.assertTrue((tss.extract().values() == target.values()).all())

    def test__TimeShop__replace_with_end_time(self):
        """

        Integration test for workflow select -> flatten -> replace

        """
        # create target values
        df = pd.DataFrame(data={'First': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]})
        inds = [f'2021-01-{day}' for day in range(1, 11)]
        df.index = pd.to_datetime(inds)
        target = TimeSeriesDarts.from_dataframe(df)

        # create TimeShop object
        tss = self.ts.edit()
        # select into the clipboard
        tss.select(other=tss.time_series, start_time=self.ts.start_time(), end_time=self.ts.end_time())
        # create values to add (this assumes that TimeShop.flatten() works)
        tss.flatten(value=2)
        # replace the target to the object to be tested
        tss.replace()

        # test that all values are the same
        self.assertTrue((tss.extract().values() == target.values()).all())

    def test__TimeShop__insert_with_end_time(self):
        """

        Integration test for workflow select -> flatten -> insert

        """
        # create initial values
        df = pd.DataFrame(data={'First': [2, 2, 2]})
        inds = [f'2021-01-{day}' for day in range(1, 4)]
        df.index = pd.to_datetime(inds)
        ts = TimeSeriesDarts.from_dataframe(df)
        # create target values
        df = pd.DataFrame(data={'First': [2, 2, 2, 2, 2, 2]})
        inds = [f'2021-01-{day}' for day in range(1, 7)]
        df.index = pd.to_datetime(inds)
        target = TimeSeriesDarts.from_dataframe(df)

        tss = ts.edit()
        # select into the clipboard
        tss.select(other=tss.time_series, start_time=tss.time_series.start_time(), end_time=tss.time_series.end_time())
        # create values to add (this assumes that TimeShop.flatten() works)
        tss.flatten(value=2)
        # replace the target to the object to be tested
        tss.insert()
        # test that all values are the same
        self.assertTrue((tss.extract().values() == target.values()).all())

    def test__TimeShop__crop_with_n_values(self):
        """

        Unittest test for the crop function

        """
        df = pd.DataFrame(data={'First': [1, 1, 1, 1, 1, 1, 1]})
        inds = [f'2021-01-{day}' for day in range(1, 8)]
        df.index = pd.to_datetime(inds)
        target = TimeSeriesDarts.from_dataframe(df)

        tss = self.ts.edit()
        # replace the target to the object to be tested
        tss.crop(start_time=self.ts.start_time(), n_values=3)

        # test that all values are the same
        self.assertTrue((tss.extract().values() == target.values()).all())

    def test__TimeShop__flatten_with_n_values(self):
        """

        Integration test for workflow select -> flatten

        """
        df = pd.DataFrame(data={'First': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]})
        inds = [f'2021-01-{day}' for day in range(1, 11)]
        df.index = pd.to_datetime(inds)
        target = TimeSeriesDarts.from_dataframe(df)

        tss = self.ts.edit()
        # select into the clipboard
        tss.select(other=tss.time_series, start_time=tss.time_series.start_time(), end_time=tss.time_series.end_time())
        # replace the target to the object to be tested
        tss.flatten(value=1)

        # test that all values are the same
        for clip in tss.clipboard:
            self.assertTrue((clip.values() == target.values()).all())

    def test__TimeShop__create_white_noise_randomness(self):
        tss = self.ts.edit()

        tss.select(other=tss.time_series, start_time=tss.time_series.start_time(), end_time=tss.time_series.end_time())
        tss.create_white_noise(mu=0, sigma=1)
        try1 = tss.clipboard

        tss.clean_clipboard()

        tss.select(other=tss.time_series, start_time=tss.time_series.start_time(), end_time=tss.time_series.end_time())
        tss.create_white_noise(mu=0, sigma=1)
        try2 = tss.clipboard

        for ind, clip in enumerate(try1):
            self.assertTrue((clip.values() != try2[ind].values()).all())

    def test__TimeShop__create_trend_with_n_values(self):
        """

        Integration test for workflow select -> create_trend

        """
        df = pd.DataFrame(data={'First': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]})
        inds = [f'2021-01-{day}' for day in range(1, 11)]
        df.index = pd.to_datetime(inds)
        target = TimeSeriesDarts.from_dataframe(df)

        tss = self.ts.edit()
        # select into the clipboard
        tss.select(other=tss.time_series, start_time=tss.time_series.start_time(), end_time=tss.time_series.end_time())
        # replace the target to the object to be tested
        tss.create_trend(slope=1)

        # test that all values are the same
        for clip in tss.clipboard:
            self.assertTrue((clip.values() == target.values()).all())

    def test__TimeShop__select(self):
        """

        Unit-test test for workflow select

        """
        df = pd.DataFrame(data={'First': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]})
        inds = [f'2021-01-{day}' for day in range(1, 11)]
        df.index = pd.to_datetime(inds)
        target = TimeSeriesDarts.from_dataframe(df)

        tss = self.ts.edit()
        # replace the target to the object to be tested
        tss.select(other=self.ts, start_time=self.ts.start_time(), end_time=self.ts.end_time())

        # test that all values are the same
        for clip in tss.clipboard:
            self.assertTrue((clip.values() == target.values()).all())

    def test__TimeShop__spiking_single_value(self):
        """

        Integration test for workflow select -> spiking -> add

        """
        df = pd.DataFrame(data={'First': [1, 1, 1, 1, 11, 1, 1, 1, 1, 1]})
        inds = [f'2021-01-{day}' for day in range(1, 11)]
        df.index = pd.to_datetime(inds)
        target = TimeSeriesDarts.from_dataframe(df)

        tss = self.ts.edit()
        # replace the target to the object to be tested
        tss.select(other=self.ts, start_time='2021-01-05', end_time='2021-01-05')
        # create spike
        tss.spiking(spike_value=10, mode='lin')
        tss.add()

        self.assertTrue((tss.extract().values() == target.values()).all())

    def test__TimeShop__threshold_search_above_value(self):
        """

        Unit-test test for select with threshold above

        """
        # create initial values
        df = pd.DataFrame(data={'First': [1, 1, 2, 2, 2, 1, 2, 2, 2, 1]})
        inds = [f'2021-01-{day}' for day in range(1, 11)]
        df.index = pd.to_datetime(inds)
        ts = TimeSeriesDarts.from_dataframe(df)

        # create target
        df = pd.DataFrame(data={'First': [2, 2, 2]})
        inds = [f'2021-01-{day}' for day in range(3, 6)]
        df.index = pd.to_datetime(inds)
        t1 = TimeSeriesDarts.from_dataframe(df)

        df = pd.DataFrame(data={'Second': [2, 2, 2]})
        inds = [f'2021-01-{day}' for day in range(7, 10)]
        df.index = pd.to_datetime(inds)
        t2 = TimeSeriesDarts.from_dataframe(df)
        target = [t1, t2]

        # create the editor
        tss = ts.edit()
        # search for the values above
        tss.threshold_search(threshold=1, operator=">")

        self.assertTrue(len(tss.clipboard) == len(target))

        for i, v in enumerate(tss.clipboard):
            self.assertTrue((v.values() == target[i].values()).all())

    def test__TimeShop__threshold_search_below_value(self):
        """

        Unit-test test for select with threshold below

        """
        df = pd.DataFrame(data={'First': [2, 2, 1, 1, 1, 2, 1, 1, 1, 2]})
        inds = [f'2021-01-{day}' for day in range(1, 11)]
        df.index = pd.to_datetime(inds)
        ts = TimeSeriesDarts.from_dataframe(df)

        # create target
        df = pd.DataFrame(data={'First': [1, 1, 1]})
        inds = [f'2021-01-{day}' for day in range(3, 6)]
        df.index = pd.to_datetime(inds)
        t1 = TimeSeriesDarts.from_dataframe(df)

        df = pd.DataFrame(data={'Second': [1, 1, 1]})
        inds = [f'2021-01-{day}' for day in range(7, 10)]
        df.index = pd.to_datetime(inds)
        t2 = TimeSeriesDarts.from_dataframe(df)
        # final target
        target = [t1, t2]

        # create the editor
        tss = ts.edit()
        # search for the values above
        tss.threshold_search(threshold=2, operator="<")

        self.assertTrue(len(tss.clipboard) == len(target))

        for i, v in enumerate(tss.clipboard):
            self.assertTrue((v.values() == target[i].values()).all())

    def test__TimeShop_time_shift(self):
        """

        Unit-test test for shift function

        """
        df = pd.DataFrame(data={'First': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]})
        inds = [f'2021-01-{day}' for day in range(11, 21)]
        df.index = pd.to_datetime(inds)
        target = TimeSeriesDarts.from_dataframe(df)

        tss = self.ts.edit()
        # select into the clipboard
        tss.select(other=tss.time_series, start_time=tss.time_series.start_time(), end_time=tss.time_series.end_time())
        # shifting the timestamps
        tss.time_shifting(new_start='2021-01-11')

        for clip in tss.clipboard:
            self.assertTrue((clip.values() == target.values()).all())

    def test__TimeShop_select_random(self):
        """

        Unit-test test for select_random function

        """
        df = pd.DataFrame(data={'First': [1, 1, 1]})
        inds = [f'2021-01-{day}' for day in range(1, 4)]
        df.index = pd.to_datetime(inds)
        target = TimeSeriesDarts.from_dataframe(df)

        tss = self.ts.edit()
        tss.select_random(length=len(target))

        for clip in tss.clipboard:
            self.assertTrue(len(clip) == len(target))
