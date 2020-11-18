from unittest import TestCase

from pandas import Index

from timeatlas.time_series.component import Component
from timeatlas.time_series.component_handler import ComponentHandler


class TestComponentHandler(TestCase):

    def test__init__is_instance(self):
        # object
        ch = ComponentHandler()
        # test
        self.assertIsInstance(ch, ComponentHandler)

    def test__init__without_arg(self):
        # object
        ch = ComponentHandler()
        # test
        self.assertIsInstance(ch.components, list)
        self.assertEqual(ch.components, [])
        self.assertEqual(len(ch.components), 0)

    def test__init__with_component_as_arg(self):
        # object
        c = Component("label")
        ch = ComponentHandler(c)
        # test
        self.assertIsInstance(ch.components, list)
        self.assertTrue(len(ch.components), 1)
        for i in ch.components:
            self.assertIsInstance(i, Component)

    def test__init__with_list_of_component_as_arg(self):
        # object
        c1 = Component("temperature")
        c2 = Component("pressure")
        component_list = [c1, c2]
        ch = ComponentHandler(component_list)
        # test
        self.assertIsInstance(ch.components, list)
        self.assertTrue(len(ch.components), 2)
        for i in ch.components:
            self.assertIsInstance(i, Component)

    def test__getitem__with_int(self):
        # object
        args1 = ["temperature", "ci-lower", "ci-upper"]
        args2 = ["pressure", "label"]
        c1 = Component(args1[0])
        c1.add_meta(args1[1])
        c1.add_meta(args1[2])
        c2 = Component(args2[0])
        c2.add_meta(args2[1])
        component_list = [c1, c2]
        ch = ComponentHandler(component_list)
        # execute function
        ch = ch[0]
        # test return type
        self.assertIsInstance(ch, ComponentHandler)
        # test content
        wanted_cols_0 = [f"0_{args1[0]}", f"0-0_{args1[1]}", f"0-1_{args1[2]}"]
        cols_0 = ch.get_columns().to_list()
        self.assertEqual(wanted_cols_0, cols_0)

    def test__getitem__with_str(self):
        # object
        args1 = ["temperature", "ci-lower", "ci-upper"]
        args2 = ["pressure", "label"]
        c1 = Component(args1[0])
        c1.add_meta(args1[1])
        c1.add_meta(args1[2])
        c2 = Component(args2[0])
        c2.add_meta(args2[1])
        component_list = [c1, c2]
        ch = ComponentHandler(component_list)
        # execute function
        ch = ch[f"0_{args1[0]}"]
        # test return type
        self.assertIsInstance(ch[0], ComponentHandler)
        # test content
        wanted_cols_0 = [f"0_{args1[0]}", f"0-0_{args1[1]}", f"0-1_{args1[2]}"]
        cols_0 = ch.get_columns().to_list()
        self.assertEqual(wanted_cols_0, cols_0)

    def test__getitem__with_list_of_int(self):
        # object
        args1 = ["temperature", "ci-lower", "ci-upper"]
        args2 = ["pressure", "label"]
        c1 = Component(args1[0])
        c1.add_meta(args1[1])
        c1.add_meta(args1[2])
        c2 = Component(args2[0])
        c2.add_meta(args2[1])
        component_list = [c1, c2]
        ch = ComponentHandler(component_list)
        # execute function
        ch = ch[[0, 1]]
        # test return type
        self.assertIsInstance(ch[0], ComponentHandler)
        # test content
        wanted_cols = [f"0_{args1[0]}", f"0-0_{args1[1]}", f"0-1_{args1[2]}",
                       f"1_{args2[0]}", f"1-0_{args2[1]}"]
        cols = ch.get_columns().to_list()
        self.assertEqual(wanted_cols, cols)

    def test__getitem__with_list_of_str(self):
        # object
        args1 = ["temperature", "ci-lower", "ci-upper"]
        args2 = ["pressure", "label"]
        c1 = Component(args1[0])
        c1.add_meta(args1[1])
        c1.add_meta(args1[2])
        c2 = Component(args2[0])
        c2.add_meta(args2[1])
        component_list = [c1, c2]
        ch = ComponentHandler(component_list)
        # execute function
        ch = ch[[f"0_{args1[0]}", f"1_{args2[0]}"]]
        # test return type
        self.assertIsInstance(ch[0], ComponentHandler)
        # test content
        wanted_cols = [f"0_{args1[0]}", f"0-0_{args1[1]}", f"0-1_{args1[2]}",
                       f"1_{args2[0]}", f"1-0_{args2[1]}"]
        cols = ch.get_columns().to_list()
        self.assertEqual(wanted_cols, cols)

    def test__delitem__with_int(self):
        # object
        args1 = ["temperature", "ci-lower", "ci-upper"]
        args2 = ["pressure", "label"]
        c1 = Component(args1[0])
        c1.add_meta(args1[1])
        c1.add_meta(args1[2])
        c2 = Component(args2[0])
        c2.add_meta(args2[1])
        component_list = [c1, c2]
        ch = ComponentHandler(component_list)
        # execute function
        len_before = len(ch)
        del ch[0]
        len_after = len(ch)
        # test
        self.assertTrue(len_before == len_after + 1)

    def test__delitem__with_str(self):
        # object
        args1 = ["temperature", "ci-lower", "ci-upper"]
        args2 = ["pressure", "label"]
        c1 = Component(args1[0])
        c1.add_meta(args1[1])
        c1.add_meta(args1[2])
        c2 = Component(args2[0])
        c2.add_meta(args2[1])
        component_list = [c1, c2]
        ch = ComponentHandler(component_list)
        # execute function
        len_before = len(ch)
        del ch["0_temperature"]
        len_after = len(ch)
        # test
        self.assertTrue(len_before == len_after + 1)

    def test__str(self):
        # object
        args1 = ["temperature", "ci-lower", "ci-upper"]
        args2 = ["pressure", "label"]
        c1 = Component(args1[0])
        c1.add_meta(args1[1])
        c1.add_meta(args1[2])
        c2 = Component(args2[0])
        c2.add_meta(args2[1])
        component_list = [c1, c2]
        ch = ComponentHandler(component_list)
        # test
        my_str = ch.__str__()
        wanted_str = str([f"0_{args1[0]}", f"0-0_{args1[1]}", f"0-1_{args1[2]}",
                          f"1_{args2[0]}", f"1-0_{args2[1]}"])
        self.assertEqual(wanted_str, my_str)

    def test__append__contains_component(self):
        # object
        c = Component("temperature")
        ch = ComponentHandler()
        # test
        self.assertEqual(len(ch.components), 0)
        ch.append(c)
        self.assertEqual(len(ch.components), 1)
        self.assertIsInstance(ch.components[0], Component)
        self.assertEqual(id(ch.components[0]), id(c))

    def test__get_component_by_name(self):
        # object
        args1 = ["temperature", "ci-lower", "ci-upper"]
        args2 = ["pressure", "label"]
        c1 = Component(args1[0])
        c1.add_meta(args1[1])
        c1.add_meta(args1[2])
        c2 = Component(args2[0])
        c2.add_meta(args2[1])
        component_list = [c1, c2]
        ch = ComponentHandler(component_list)
        # test
        c = ch.get_component_by_name("0_temperature")

    def test__get_column_by_id__gives_right_column_names(self):
        # object
        args1 = ["temperature", "ci-lower", "ci-upper"]
        args2 = ["pressure", "label"]
        c1 = Component(args1[0])
        c1.add_meta(args1[1])
        c1.add_meta(args1[2])
        c2 = Component(args2[0])
        c2.add_meta(args2[1])
        component_list = [c1, c2]
        ch = ComponentHandler(component_list)
        # test first component
        wanted_cols_0 = [f"0_{args1[0]}", f"0-0_{args1[1]}", f"0-1_{args1[2]}"]
        cols_0 = ch.get_column_by_id(0).to_list()
        self.assertEqual(wanted_cols_0, cols_0)
        # test second component
        wanted_cols_1 = [f"1_{args2[0]}", f"1-0_{args2[1]}"]
        cols_1 = ch.get_column_by_id(1).to_list()
        self.assertEqual(wanted_cols_1, cols_1)

    def test__get_column_by_id__without_meta(self):
        # object
        args1 = ["temperature", "ci-lower", "ci-upper"]
        args2 = ["pressure", "label"]
        c1 = Component(args1[0])
        c1.add_meta(args1[1])
        c1.add_meta(args1[2])
        c2 = Component(args2[0])
        c2.add_meta(args2[1])
        component_list = [c1, c2]
        ch = ComponentHandler(component_list)
        # test 1 - cols are not equal
        wanted_cols_0 = [f"0_{args1[0]}", f"0-0_{args1[1]}", f"0-1_{args1[2]}"]
        cols_0 = ch.get_column_by_id(0, with_meta=False).to_list()
        self.assertNotEqual(wanted_cols_0, cols_0)
        # test 2 - cols are equal
        wanted_cols_1 = [f"1_{args2[0]}"]
        cols_1 = ch.get_column_by_id(1, with_meta=False).to_list()
        self.assertEqual(wanted_cols_1, cols_1)

    def test__get_column_by_id__return_pandas_index(self):
        # object
        args1 = ["temperature", "ci-lower", "ci-upper"]
        args2 = ["pressure", "label"]
        c1 = Component(args1[0])
        c1.add_meta(args1[1])
        c1.add_meta(args1[2])
        c2 = Component(args2[0])
        c2.add_meta(args2[1])
        component_list = [c1, c2]
        ch = ComponentHandler(component_list)
        # test
        cols_0 = ch.get_column_by_id(0)
        self.assertIsInstance(cols_0, Index)

    def test__get_column_by_name__gives_right_columms_names(self):
        # object
        args1 = ["temperature", "ci-lower", "ci-upper"]
        args2 = ["pressure", "label"]
        c1 = Component(args1[0])
        c1.add_meta(args1[1])
        c1.add_meta(args1[2])
        c2 = Component(args2[0])
        c2.add_meta(args2[1])
        component_list = [c1, c2]
        ch = ComponentHandler(component_list)
        # test 1
        wanted_cols_0 = [f"0_{args1[0]}", f"0-0_{args1[1]}", f"0-1_{args1[2]}"]
        cols_0 = ch.get_column_by_name(f"0_{args1[0]}").to_list()
        self.assertEqual(wanted_cols_0, cols_0)
        # test 2
        wanted_cols_1 = [f"1_{args2[0]}", f"1-0_{args2[1]}"]
        cols_1 = ch.get_column_by_name(f"1_{args2[0]}").to_list()
        self.assertEqual(wanted_cols_1, cols_1)

    def test__get_columns__gives_right_column_names(self):
        # object
        args1 = ["temperature", "ci-lower", "ci-upper"]
        args2 = ["pressure", "label"]
        c1 = Component(args1[0])
        c1.add_meta(args1[1])
        c1.add_meta(args1[2])
        c2 = Component(args2[0])
        c2.add_meta(args2[1])
        component_list = [c1, c2]
        ch = ComponentHandler(component_list)
        # test
        wanted_cols = [f"0_{args1[0]}", f"0-0_{args1[1]}", f"0-1_{args1[2]}",
                       f"1_{args2[0]}", f"1-0_{args2[1]}"]
        cols = ch.get_columns().to_list()
        self.assertEqual(wanted_cols, cols)

    def test__get_columns__without_meta(self):
        # object
        args1 = ["temperature", "ci-lower", "ci-upper"]
        args2 = ["pressure", "label"]
        c1 = Component(args1[0])
        c1.add_meta(args1[1])
        c1.add_meta(args1[2])
        c2 = Component(args2[0])
        c2.add_meta(args2[1])
        component_list = [c1, c2]
        ch = ComponentHandler(component_list)
        # test 1 - cols are not equal
        wanted_cols_1 = [f"0_{args1[0]}", f"0-0_{args1[1]}", f"0-1_{args1[2]}",
                         f"1_{args2[0]}", f"1-0_{args2[1]}"]
        cols_1 = ch.get_columns(with_meta=False).to_list()
        self.assertNotEqual(wanted_cols_1, cols_1)
        # test 2 - cols are equal
        wanted_cols_2 = [f"0_{args1[0]}", f"1_{args2[0]}"]
        cols_2 = ch.get_columns(with_meta=False).to_list()
        self.assertEqual(wanted_cols_2, cols_2)

    def test__get_columns__returns_pandas_index(self):
        # object
        args1 = ["temperature", "ci-lower", "ci-upper"]
        args2 = ["pressure", "label"]
        c1 = Component(args1[0])
        c1.add_meta(args1[1])
        c1.add_meta(args1[2])
        c2 = Component(args2[0])
        c2.add_meta(args2[1])
        component_list = [c1, c2]
        ch = ComponentHandler(component_list)
        # test
        cols = ch.get_columns()
        self.assertIsInstance(cols, Index)

    def test__copy__returns_different_id(self):
        # object
        args1 = ["temperature", "ci-lower", "ci-upper"]
        args2 = ["pressure", "label"]
        c1 = Component(args1[0])
        c1.add_meta(args1[1])
        c1.add_meta(args1[2])
        c2 = Component(args2[0])
        c2.add_meta(args2[1])
        component_list = [c1, c2]
        ch = ComponentHandler(component_list)
        # test
        deepcopy = ch.copy()
        self.assertNotEqual(id(ch.components), id(deepcopy.components))
        # test
        copy = ch.copy(deep=False)
        self.assertEqual(id(ch.components), id(copy.components))

