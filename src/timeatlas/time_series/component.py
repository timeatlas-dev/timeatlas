from typing import NoReturn

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
        self.name = self.series[COMPONENT_VALUES]

        # Add metadata if present
        if metadata is not None:
            self.metadata = metadata
        else:
            self.metadata = None

    def get_main(self) -> str:
        """ Get the name of the main series in the component

        Returns:
            str of the name of the main series
        """
        return self.series[COMPONENT_VALUES]

    def get_all(self) -> list:
        """ Get the names of all series in this component

        Returns:
            list with all the series names
        """
        return list(self.series.values())
