import unittest
from src.timeatlas import handle
from src.timeatlas.data import *
import pandas as pd
from typing import Dict


class TestHandler(unittest.TestCase):

    def setUp(self) -> None:
        self.handler = handle.Handle()

    def test__load_from_csv(self):
        path = "./data/raw/dummy"
        ds = self.handler.load_from_csv(path)
        self.__check_dataset_types(ds)

    def test__load_from_bbdata(self):
        obj_ids = [2902, 2920]
        from_timestamp = "2018-01-01T00:00"
        to_timestamp = "2018-02-01T00:00"
        ds = self.handler.load_from_bbdata(obj_ids, from_timestamp, to_timestamp)
        self.__check_dataset_types(ds)

    def test__export(self):
        path = "./data/raw/dummy"
        ds = self.handler.load_from_csv(path)
        self.__check_dataset_types(ds)
        self.handler.export(ds, "./data/raw/test/")

    def __check_dataset_types(self, ds):
        self.assertIsInstance(ds, Dataset)
        self.assertIsInstance(ds.values, Dict)
        self.assertIsInstance(ds.objects, pd.DataFrame)