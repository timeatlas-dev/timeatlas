from typing import NoReturn

from timeatlas.abstract import AbstractBaseDetector


class Surprise(AbstractBaseDetector):

    def __init__(self, model, error_func, warning: float = 0.85, critical: float = 0.95):
        super(self).__init__()

    def detect(self) -> NoReturn:
        pass