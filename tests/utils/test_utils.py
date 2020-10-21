from unittest import TestCase
import os
import shutil

from timeatlas.utils import ensure_dir, to_pickle


class TestUtils(TestCase):

    def test__Utils__ensure_dir(self):
        # setup test
        dir_path = './ensure/'
        ensure_dir(dir_path)

        # test
        self.assertTrue(os.path.exists(dir_path))
        self.assertTrue(os.path.isdir(dir_path))

        # clean up
        os.rmdir(dir_path)
        self.assertFalse(os.path.isdir(dir_path))

    def test__Utils__to_pickle(self):
        out_dir = '../data/test-import/util_to_pickle/'

        data = 5

        to_pickle(obj=data, path=f"{out_dir}/pickle_test.pkl")

        self.assertTrue(os.path.isfile(f"{out_dir}/pickle_test.pkl"))

        # clean up
        shutil.rmtree(out_dir)
        # check if cleaned
        self.assertFalse(os.path.isdir(out_dir))
