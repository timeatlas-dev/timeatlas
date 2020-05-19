from typing import List
from unittest import TestCase
from timeatlas.metadata import Metadata
from timeatlas.types import *


class TestMetadata(TestCase):

    def setUp(self) -> None:

        # known type in dict
        self.my_unit = {
            "name": "power",
            "symbol": "W",
            "data_type": "float"
        }

        # known type in dedicated object
        self.my_sensor = Sensor(2902, "HB/floor2/22-23C/Prises_Power_Tot")

        # random dicts
        self.my_location = {
            "building": "Blue Factory",
            "floor": "12",
            "room": "22C"
        }

        self.my_coordinates = {
            "lat": 46.796611,
            "lon": 7.147563
        }

        self.my_dict = {
            "unit": self.my_unit,
            "sensor": self.my_sensor,
            "location": self.my_location,
            "coordinates": self.my_coordinates
        }

    def test_construct(self):
        my_metadata = Metadata()
        self.assertIsInstance(my_metadata, Metadata)

    def test_construct_with_dict(self):
        my_metadata = Metadata(self.my_dict)
        self.assertIsInstance(my_metadata, Metadata)

    def test_add_known_type_dict(self):
        my_metadata = Metadata()
        my_known_dict = {
            "unit": self.my_unit
        }
        my_metadata.add(my_known_dict)
        self.assertTrue("unit" in list(my_metadata.keys()),
                        "'unit' has not been added to the metadata")
        self.assertIsInstance(my_metadata["unit"], Unit)

    def test_add_known_type_object(self):
        my_metadata = Metadata()
        my_known_obj = {
            "sensor": self.my_sensor
        }
        my_metadata.add(my_known_obj)
        self.assertTrue("sensor" in list(my_metadata.keys()),
                        "'sensor' has not been added to the metadata")
        self.assertIsInstance(my_metadata["sensor"], Sensor)

    def test_add_dict(self):
        my_metadata = Metadata()
        my_dict = {
            "coordinates": self.my_coordinates
        }
        my_metadata.add(my_dict)
        self.assertTrue("coordinates" in list(my_metadata.keys()),
                        "'coordinates' has not been added to the metadata")
