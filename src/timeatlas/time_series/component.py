from typing import NoReturn

import pandas as pd

from timeatlas.metadata import Metadata
from timeatlas.config.constants import COMPONENT_VALUES, COMPONENT_META_PREFIX


class Component:
    """ Component of a Time Series

    This class stores information about the component of a time series:

      - the name of the main series
      - the names of all meta series
      - metadata about the component
    """

    def __init__(self, name: str, metadata: Metadata = None):

        self.series = {}  # Component series storage

        # Create the main value of the component
        assert name is not None, "name argument can't be None"
        self.series[COMPONENT_VALUES] = name

        self.n_meta = 0  # Initialize the meta series

        # Add metadata if present
        if metadata is not None:
            self.metadata = metadata
        else:
            self.metadata = None

    def add_meta_series(self, name: str) -> NoReturn:
        """ Add a meta series to the Component

        Args:
            name: str giving the name to the meta series
        """
        self.series[f"{COMPONENT_META_PREFIX}{self.n_meta}"] = \
            f"{self.n_meta}_{name}"
        self.n_meta = self.n_meta + 1

    def get_columns(self) -> pd.Index:
        """ Gives the column names of this component as if they were in a Pandas
        DataFrame.

        Allows for easy selection of the data stored in a DataFrame when used
        in association with a ComponentHandler

        Returns:
            Pandas Index
        """
        return pd.Index(list(self.series.values()))
