from unittest import TestCase
from timeatlas.data_structures.unit import Unit


class TestUnit(TestCase):

    def setUp(self) -> None:
        self.my_unit = Unit("Temperature", "°C", "float")

    def test__Unit__has_right_types(self):
        self.assertTrue(type(self.my_unit.name) is str)
        self.assertTrue(type(self.my_unit.symbol) is str)
        self.assertTrue(type(self.my_unit.data_type) is str)

    def test__Unit__is_instance(self):
        self.assertIsInstance(self.my_unit, Unit)
