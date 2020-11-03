from unittest import TestCase

from timeatlas.time_series.component import Component
from timeatlas.metadata import Metadata
from timeatlas.config.constants import *


class TestComponent(TestCase):

    def test__init__is_instance(self):
        c = Component("test")
        self.assertIsInstance(c, Component)

    def test__init__throw_type_error_without_name(self):
        with self.assertRaises(TypeError):
            Component()

    def test__init__name_is_correct(self):
        name_arg = "test"
        c = Component(name_arg)
        name = c.series[COMPONENT_VALUES]
        self.assertEqual(name_arg, name)

    def test__init__metadata_can_be_added(self):
        # test the absence of Metadata
        c1 = Component("test")
        self.assertTrue(c1.metadata is None)
        # test the presence of Metadata
        m = Metadata()
        c2 = Component("test", m)
        self.assertTrue(c2.metadata is not None)
        self.assertEqual(id(m), id(c2.metadata))

    def test__add_meta_series__should_increment_n_meta(self):
        c = Component("test")
        self.assertEqual(0, c.n_meta)
        c.add_meta("ci_lower")
        self.assertEqual(1, c.n_meta)
        c.add_meta("ci_upper")
        self.assertEqual(2, c.n_meta)

    def test__add_meta_series__series_dict_has_right_items(self):
        # prepare object
        args = ["test", "ci_lower", "ci_upper"]
        c = Component(args[0])
        c.add_meta(args[1])
        c.add_meta(args[2])

        # test keys
        k = list(c.series.keys())
        wanted_k = [COMPONENT_VALUES,
                    f"{COMPONENT_META_PREFIX}0",
                    f"{COMPONENT_META_PREFIX}1"]
        self.assertEqual(k, wanted_k)

        # test values
        v = list(c.series.values())
        wanted_v = [args[0],
                    f"0_{args[1]}",
                    f"1_{args[2]}"]
        self.assertEqual(v, wanted_v)

    def test__get_main__return_list(self):
        # prepare object
        args = ["test", "ci_lower", "ci_upper"]
        c = Component(args[0])
        c.add_meta(args[1])
        c.add_meta(args[2])

        # test
        cols = c.get_main()
        self.assertIsInstance(cols, list)

    def test__get_main__has_right_elements(self):
        # prepare object
        args = ["test", "ci_lower", "ci_upper"]
        c = Component(args[0])
        c.add_meta(args[1])
        c.add_meta(args[2])

        # test
        cols = c.get_main()
        wanted_cols = [args[0]]
        self.assertEqual(cols, wanted_cols)

    def test__get_meta__return_list(self):
        # prepare object
        args = ["test", "ci_lower", "ci_upper"]
        c = Component(args[0])
        c.add_meta(args[1])
        c.add_meta(args[2])

        # test
        cols = c.get_meta()
        self.assertIsInstance(cols, list)

    def test__get_meta__has_right_elements(self):
        # prepare object
        args = ["test", "ci_lower", "ci_upper"]
        c = Component(args[0])
        c.add_meta(args[1])
        c.add_meta(args[2])

        # test
        cols = c.get_meta()
        wanted_cols = [f"0_{args[1]}",
                       f"1_{args[2]}"]
        self.assertEqual(cols, wanted_cols)

    def test__get_all__return_list(self):
        # prepare object
        args = ["test", "ci_lower", "ci_upper"]
        c = Component(args[0])
        c.add_meta(args[1])
        c.add_meta(args[2])

        # test
        cols = c.get_all()
        self.assertIsInstance(cols, list)

    def test__get_all__has_right_elements(self):
        # prepare object
        args = ["test", "ci_lower", "ci_upper"]
        c = Component(args[0])
        c.add_meta(args[1])
        c.add_meta(args[2])

        # test
        cols = c.get_all()
        wanted_cols = [f"{args[0]}",
                       f"0_{args[1]}",
                       f"1_{args[2]}"]
        self.assertEqual(cols, wanted_cols)