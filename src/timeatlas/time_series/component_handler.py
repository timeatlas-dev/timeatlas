from copy import deepcopy, copy

import pandas as pd

from .component import Component
from timeatlas.config.constants import TIME_SERIES_VALUES


class ComponentHandler(list):

    def __init__(self):
        super().__init__(self)

    def append(self, component: Component):
        super().append(component)

    def get_component_columns(self, i):
        cols = []
        for k, v in self[i].items():
            col_name = self.__format_value_str(i, v) \
                if k == TIME_SERIES_VALUES \
                else self.__format_meta_str(i, v)
            cols.append(col_name)
        return cols

    def get_columns(self):
        cols = []
        for i, c in enumerate(self):
            cols += self.get_component_columns(i)
        return pd.Index(cols)

    def get_values(self):
        values = []
        for i, c in enumerate(self):
            values.append(self.__format_value_str(i, c[TIME_SERIES_VALUES]))
        return values

    def copy(self, deep=False) -> 'ComponentHandler':
        return deepcopy(self) if deep else copy(self)

    @staticmethod
    def __format_value_str(index, value):
        return f"{index}_{value}"

    @staticmethod
    def __format_meta_str(index, value):
        return f"{index}-{value}"
