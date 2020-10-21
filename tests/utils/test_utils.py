from unittest import TestCase
from os import path, rmdir

from timeatlas.utils import ensure_dir, to_pickle


class TestUtils(TestCase):

    def test__Utils__ensure_dir(self):
        # setup test
        dir_path = './ensure/'
        ensure_dir(dir_path)

        # test
        self.assertTrue(path.exists(dir_path))
        self.assertTrue(path.isdir(dir_path))

        # clean up
        rmdir(dir_path)