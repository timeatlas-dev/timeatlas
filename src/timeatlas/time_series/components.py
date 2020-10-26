import pandas as pd

from .component import Component
from timeatlas.config.constants import TIME_SERIES_VALUES


class Components(list):

    def __init__(self):
        super().__init__(self)

    def append(self, name: str):
        c = Component(name)
        super().append(c)

    def get_columns(self):
        cols = []
        for i, c in enumerate(self):
            for k, v in c.items():
                col_name = f"{i}_{v}" if k == TIME_SERIES_VALUES else f"{i}-{v}"
                cols.append(col_name)
        return pd.Index(cols)
