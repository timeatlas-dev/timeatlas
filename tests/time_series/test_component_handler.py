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

    def test__get_component_by_id__gives_right_column_names(self):
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
        cols_0 = ch.get_component_by_id(0).to_list()
        self.assertEqual(wanted_cols_0, cols_0)
        # test second component
        wanted_cols_1 = [f"1_{args2[0]}", f"1-0_{args2[1]}"]
        cols_1 = ch.get_component_by_id(1).to_list()
        self.assertEqual(wanted_cols_1, cols_1)

    def test__get_component_by_id__without_meta(self):
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
        cols_0 = ch.get_component_by_id(0, with_meta=False).to_list()
        self.assertNotEqual(wanted_cols_0, cols_0)
        # test 2 - cols are equal
        wanted_cols_1 = [f"1_{args2[0]}"]
        cols_1 = ch.get_component_by_id(1, with_meta=False).to_list()
        self.assertEqual(wanted_cols_1, cols_1)

    def test__get_component_by_id__return_pandas_index(self):
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
        cols_0 = ch.get_component_by_id(0)
        self.assertIsInstance(cols_0, Index)

    def test__get_components__gives_right_column_names(self):
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
        cols = ch.get_components().to_list()
        self.assertEqual(wanted_cols, cols)

    def test__get_components__returns_pandas_index(self):
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
        cols = ch.get_components()
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
