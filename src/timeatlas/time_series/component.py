import pandas as pd

from timeatlas.metadata import Metadata
from timeatlas.config.constants import TIME_SERIES_VALUES


class Component(dict):

    def __init__(self, name: str, metadata: Metadata = None):
        super().__init__(self)
        self[TIME_SERIES_VALUES] = name
        self.n_meta = None
        if metadata is not None:
            self.metadata = metadata
        else:
            self.metadata = None

    def add(self, value):
        self.n_meta = self.n_meta + 1 if type(self.n_meta) is int else 0
        self[f"meta_{self.n_meta}"] = f"{self.n_meta}_{value}"

    def get_columns(self):
        return pd.Index(list(self.values()))
