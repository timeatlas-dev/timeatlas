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
        # create values to add (this assumes that TimeShop.flat() works)
        tss.flat(start_time=self.ts.start_time(), end_time=self.ts.end_time(), value=1)
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
        # create values to add (this assumes that TimeShop.flat() works)
        tss.flat(start_time=self.ts.start_time(), end_time=self.ts.end_time(), value=2)
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
        # create values to add (this assumes that TimeShop.flat() works)
        tss.flat(start_time=self.ts.start_time(), end_time=self.ts.end_time(), value=2)
        # replace the target to the object to be tested
        tss.replace()

        # test that all values are the same
        self.assertTrue((tss.extract().values() == target.values()).all())

    def test__TimeShop__insert_with_n_values(self):
        # create initial values
        df = pd.DataFrame(data={'First': [2, 2, 2]})
        inds = [f'2021-01-{day}' for day in range(1, 4)]
        df.index = pd.to_datetime(inds)
        ts = TimeSeriesDarts.from_dataframe(df)
        # create target values
        df = pd.DataFrame(data={'First': [2, 2, 2, 2, 2]})
        inds = [f'2021-01-{day}' for day in range(1, 6)]
        df.index = pd.to_datetime(inds)
        target = TimeSeriesDarts.from_dataframe(df)

        tss = ts.edit()
        # create values to add (this assumes that TimeShop.flat() works)
        tss.flat(start_time=ts.start_time(), n_values=2, value=2)
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
        df = pd.DataFrame(data={'First': [1, 1, 1, 1, 1, 1, 1]})
        inds = [f'2021-01-{day}' for day in range(1, 8)]
        df.index = pd.to_datetime(inds)
        target = TimeSeriesDarts.from_dataframe(df)

        tss = self.ts.edit()
        # replace the target to the object to be tested
        tss.flat(start_time=self.ts.start_time(), n_values=7, value=1)

        # test that all values are the same
        self.assertTrue((tss.generator_output.values() == target.values()).all())

    def test__TimeShop__white_noise(self):
        pass

    def test__TimeShop__trend_with_n_values(self):
        df = pd.DataFrame(data={'First': [0, 1, 2, 3, 4, 5, 6]})
        inds = [f'2021-01-{day}' for day in range(1, 8)]
        df.index = pd.to_datetime(inds)
        target = TimeSeriesDarts.from_dataframe(df)

        tss = self.ts.edit()
        # replace the target to the object to be tested
        tss.trend(slope=1, n_values=7, start_time=self.ts.start_time())

        # test that all values are the same
        self.assertTrue((tss.generator_output.values() == target.values()).all())

    def test__TimeShop__copy(self):
        df = pd.DataFrame(data={'First': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]})
        inds = [f'2021-01-{day}' for day in range(1, 11)]
        df.index = pd.to_datetime(inds)
        target = TimeSeriesDarts.from_dataframe(df)

        tss = self.ts.edit()
        # replace the target to the object to be tested
        tss.copy(other=self.ts, start_copy=self.ts.start_time(), end_copy=self.ts.end_time(),
                 insert_start=self.ts.start_time())

        # test that all values are the same
        self.assertTrue((tss.generator_output.values() == target.values()).all())

    def test__TimeShop__spike_single_value(self):
        df = pd.DataFrame(data={'First': [1, 1, 1, 1, 11, 1, 1, 1, 1, 1]})
        inds = [f'2021-01-{day}' for day in range(1, 11)]
        df.index = pd.to_datetime(inds)
        target = TimeSeriesDarts.from_dataframe(df)

        tss = self.ts.edit()
        # create spike
        tss.spike(spike_time='2021-01-05', spike_value=10, length=1, mode='lin')
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
        tss = self.ts.edit()
        # search for the values above
        tss.threshold_search(threshold=1, operator=">")
        tss.extract().values()

        self.assertTrue(len(tss.generator_output) == len(target))

        for i, v in enumerate(tss.generator_output):
            self.assertTrue((v.values() == target[i].values()).all())

    def test__TimeShop__threshold_search_below_value(self):
        df = pd.DataFrame(data={'First': [2, 2, 1, 1, 1, 2, 1, 1, 1, 2]})
        inds = [f'2021-01-{day}' for day in range(1, 11)]
        df.index = pd.to_datetime(inds)
        target = TimeSeriesDarts.from_dataframe(df)

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
        tss = self.ts.edit()
        # search for the values above
        tss.threshold_search(threshold=2, operator="<")
        tss.extract().values()

        self.assertTrue(len(tss.generator_output) == len(target))

        for i, v in enumerate(tss.generator_output):
            self.assertTrue((v.values() == target[i].values()).all())
