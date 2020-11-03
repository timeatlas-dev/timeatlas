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

    def add_meta(self, name: str) -> NoReturn:
        """ Add a meta series to the Component

        Args:
            name: str giving the name to the meta series
        """
        self.series[f"{COMPONENT_META_PREFIX}{self.n_meta}"] = \
            f"{self.n_meta}_{name}"
        self.n_meta = self.n_meta + 1

    def get_main(self) -> list:
        """ Get the name of the main series in the component

        Returns:
            list with the name of the main series
        """
        return [self.series[COMPONENT_VALUES]]

    def get_meta(self) -> list:
        """ Get the names of all meta series in this component

        Returns:
            list with all the names of the meta series
        """
        blacklist = [COMPONENT_VALUES]
        return [v for k, v in self.series.items() if k not in blacklist]

    def get_all(self) -> list:
        """ Get the names of all series in this component

        Returns:
            list with all the series names
        """
        return list(self.series.values())


