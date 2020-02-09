from unittest import TestCase
from timeatlas.data_structures import Object


class TestObject(TestCase):

    def setUp(self) -> None:
        self.my_object = Object(1, "My office sensor")

    def test__Object__has_right_types(self):
        self.assertTrue(type(self.my_object.id) is int)
        self.assertTrue(type(self.my_object.name) is str)

    def test__Object__is_instance(self):
        self.assertIsInstance(self.my_object, Object)

