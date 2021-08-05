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
        # create target values
        df = pd.DataFrame(data={'First': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]})
        inds = [f'2021-01-{day}' for day in range(1, 11)]
        df.index = pd.to_datetime(inds)
        target = TimeSeriesDarts.from_dataframe(df)

        # create TimeShop object
        tss = self.ts.edit()
        # copy into the clipboard
        tss.copy(other=tss.time_series, start_time=self.ts.start_time(), end_time=self.ts.end_time())
        # create values to add (this assumes that TimeShop.flat() works)
        tss.flat(value=1)
        # add the target to the object to be tested
        tss.add()

        # test that all values are the same
        self.assertTrue((tss.extract().values() == target.values()).all())

    def test__TimeShop__multiply_with_end_time(self):
        # create target values
        df = pd.DataFrame(data={'First': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]})
        inds = [f'2021-01-{day}' for day in range(1, 11)]
        df.index = pd.to_datetime(inds)
        target = TimeSeriesDarts.from_dataframe(df)

        # create TimeShop object
        tss = self.ts.edit()
        # copy into the clipboard
        tss.copy(other=tss.time_series, start_time=self.ts.start_time(), end_time=self.ts.end_time())
        # create values to add (this assumes that TimeShop.flat() works)
        tss.flat(value=2)
        # multiply the target to the object to be tested
        tss.multiply()

        # test that all values are the same
        self.assertTrue((tss.extract().values() == target.values()).all())

    def test__TimeShop__replace_with_end_time(self):
        # create target values
        df = pd.DataFrame(data={'First': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]})
        inds = [f'2021-01-{day}' for day in range(1, 11)]
        df.index = pd.to_datetime(inds)
        target = TimeSeriesDarts.from_dataframe(df)

        # create TimeShop object
        tss = self.ts.edit()
        # copy into the clipboard
        tss.copy(other=tss.time_series, start_time=self.ts.start_time(), end_time=self.ts.end_time())
        # create values to add (this assumes that TimeShop.flat() works)
        tss.flat(value=2)
        # replace the target to the object to be tested
        tss.replace()

        # test that all values are the same
        self.assertTrue((tss.extract().values() == target.values()).all())

    def test__TimeShop__insert_with_end_time(self):
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
        # copy into the clipboard
        tss.copy(other=tss.time_series, start_time=tss.time_series.start_time(), end_time=tss.time_series.end_time())
        # create values to add (this assumes that TimeShop.flat() works)
        tss.flat(value=2)
        # replace the target to the object to be tested
        tss.insert()
        # test that all values are the same
        self.assertTrue((tss.extract().values() == target.values()).all())

    def test__TimeShop__crop_with_n_values(self):
        df = pd.DataFrame(data={'First': [1, 1, 1, 1, 1, 1, 1]})
        inds = [f'2021-01-{day}' for day in range(1, 8)]
        df.index = pd.to_datetime(inds)
        target = TimeSeriesDarts.from_dataframe(df)

        tss = self.ts.edit()
        # replace the target to the object to be tested
        tss.crop(start_time=self.ts.start_time(), n_values=3)

        # test that all values are the same
        self.assertTrue((tss.extract().values() == target.values()).all())

    def test__TimeShop__flat_with_n_values(self):
        df = pd.DataFrame(data={'First': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]})
        inds = [f'2021-01-{day}' for day in range(1, 11)]
        df.index = pd.to_datetime(inds)
        target = TimeSeriesDarts.from_dataframe(df)

        tss = self.ts.edit()
        # copy into the clipboard
        tss.copy(other=tss.time_series, start_time=tss.time_series.start_time(), end_time=tss.time_series.end_time())
        # replace the target to the object to be tested
        tss.flat(value=1)

        # test that all values are the same
        for clip in tss.clipboard:
            self.assertTrue((clip.values() == target.values()).all())

    def test__TimeShop__white_noise(self):
        pass

    def test__TimeShop__trend_with_n_values(self):
        df = pd.DataFrame(data={'First': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]})
        inds = [f'2021-01-{day}' for day in range(1, 11)]
        df.index = pd.to_datetime(inds)
        target = TimeSeriesDarts.from_dataframe(df)

        tss = self.ts.edit()
        # copy into the clipboard
        tss.copy(other=tss.time_series, start_time=tss.time_series.start_time(), end_time=tss.time_series.end_time())
        # replace the target to the object to be tested
        tss.trend(slope=1)

        # test that all values are the same
        for clip in tss.clipboard:
            self.assertTrue((clip.values() == target.values()).all())

    def test__TimeShop__copy(self):
        df = pd.DataFrame(data={'First': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]})
        inds = [f'2021-01-{day}' for day in range(1, 11)]
        df.index = pd.to_datetime(inds)
        target = TimeSeriesDarts.from_dataframe(df)

        tss = self.ts.edit()
        # replace the target to the object to be tested
        tss.copy(other=self.ts, start_time=self.ts.start_time(), end_time=self.ts.end_time())

        # test that all values are the same
        for clip in tss.clipboard:
            self.assertTrue((clip.values() == target.values()).all())

    def test__TimeShop__spike_single_value(self):
        df = pd.DataFrame(data={'First': [1, 1, 1, 1, 11, 1, 1, 1, 1, 1]})
        inds = [f'2021-01-{day}' for day in range(1, 11)]
        df.index = pd.to_datetime(inds)
        target = TimeSeriesDarts.from_dataframe(df)

        tss = self.ts.edit()
        # replace the target to the object to be tested
        tss.copy(other=self.ts, start_time='2021-01-05', end_time='2021-01-05')
        # create spike
        tss.spike(spike_value=10, mode='lin')
        tss.add()

        self.assertTrue((tss.extract().values() == target.values()).all())

    def test__TimeShop__threshold_search_above_value(self):
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
        target = [t1, t2]

        # create the editor
        tss = ts.edit()
        # search for the values above
        tss.threshold_search(threshold=2, operator="<")

        self.assertTrue(len(tss.clipboard) == len(target))

        for i, v in enumerate(tss.clipboard):
            self.assertTrue((v.values() == target[i].values()).all())

    def test__TimeShop_shift(self):
        df = pd.DataFrame(data={'First': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]})
        inds = [f'2021-01-{day}' for day in range(11, 21)]
        df.index = pd.to_datetime(inds)
        target = TimeSeriesDarts.from_dataframe(df)

        tss = self.ts.edit()
        # copy into the clipboard
        tss.copy(other=tss.time_series, start_time=tss.time_series.start_time(), end_time=tss.time_series.end_time())
        # shifting the timestamps
        tss.shift(new_start='2021-01-11')

        for clip in tss.clipboard:
            self.assertTrue((clip.values() == target.values()).all())

    def test__TimeShop_select_random(self):
        df = pd.DataFrame(data={'First': [1, 1, 1]})
        inds = [f'2021-01-{day}' for day in range(1, 4)]
        df.index = pd.to_datetime(inds)
        target = TimeSeriesDarts.from_dataframe(df)

        tss = self.ts.edit()

        tss.random(lenght=3)

        for clip in tss.clipboard:
            self.assertTrue(len(clip) == len(target))
