from typing import NoReturn, Any
from timeatlas.utils import ensure_dir, to_pickle

from timeatlas.abstract import AbstractOutputText, AbstractOutputPickle


class IO(AbstractOutputText, AbstractOutputPickle):

    def to_text(self, path: str) -> NoReturn:
        ensure_dir(path)
        for i, ts in enumerate(self.data):
            ts_path = "{}/{}".format(path, i)
            ensure_dir(ts_path)
            ts.to_text(ts_path)

    def to_pickle(self, path: str) -> NoReturn:
        to_pickle(self, path)
