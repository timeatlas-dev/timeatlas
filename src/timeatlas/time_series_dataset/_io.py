from typing import NoReturn, Any

from timeatlas.abstract import AbstractOutputText, AbstractOutputPickle


class IO(AbstractOutputText, AbstractOutputPickle):

    def to_text(self, path: str) -> NoReturn:
        pass

    def to_pickle(self, path: str) -> NoReturn:
        pass
