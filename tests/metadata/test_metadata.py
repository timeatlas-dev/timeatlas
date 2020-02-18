from typing import List
from unittest import TestCase
from timeatlas.time_series_metadata import TimeSeriesMetadata


class TestMetadata(TestCase):

    def setUp(self) -> None:
        self.my_metadata = TimeSeriesMetadata()

    def test_add(self):
        self.my_metadata.add("name", "Room 1 / Temperature sensor")
        self.assertTrue("name" in list(self.my_metadata.list()),
                        "'name' has not been added to the time_series_metadata")

    def test_remove(self):
        self.my_metadata.add("name", "Room 1 / Temperatur sensor")
        self.my_metadata.remove("name")
        self.assertTrue("name" not in list(self.my_metadata.list()),
                        "'name' is still present in the time_series_metadata")

    def test_get(self):
        self.my_metadata.add("name", "Room 1 / Temperature sensor")
        value = self.my_metadata.get("name")
        expected_value = "Room 1 / Temperature sensor"
        self.assertTrue(value == expected_value,
                        "The stored value is not the same as the expected "
                        "value")

    def test_list(self):
        self.my_metadata.add("name", "Temperature sensor")
        self.my_metadata.add("location", "Room 1")
        keys = self.my_metadata.list()
        self.assertIsInstance(keys, List,
                              "The returned type isn't the expected one")

    def tearDown(self) -> None:
        del self.my_metadata
