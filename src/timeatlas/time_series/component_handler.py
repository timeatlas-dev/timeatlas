from typing import List, Union, NoReturn
from copy import deepcopy, copy

from pandas import Index

from .component import Component


class ComponentHandler:
    """ Helper class to manage many components

    The purpose of this class is to make the management of components in a
    time series as simple as possible, with one or many components.

    The underlying data structure is a simple list where component are stored.
    """

    def __init__(self, components: Union[List[Component], Component] = None):
        if isinstance(components, Component):
            components = [components]
        self.components = components if components is not None else []

    def append(self, component: Component) -> NoReturn:
        """
        Append a Component to the ComponentHandler

        Args:
            component: Component to append
        """
        self.components.append(component)

    def get_component_by_id(self, index: int, with_meta: bool = True) -> Index:
        """ Get Pandas Index of a Component by ID

        Get Pandas Index of a Component from the ComponentHandler by its
        positional identifier

        Args:
            index: int of the index of the component in the ComponentHandler
            with_meta: bool to include or not meta series in the return value

        Returns:
            Pandas Index of the names of the component
        """
        c = self.components[index]
        cols = self.__format_main_series(index, c.get_main())
        if with_meta:
            meta = self.__format_meta_series(index, c.get_meta())
            cols += meta
        return Index(cols)

    def get_components(self, with_meta=True) -> Index:
        """ Get Pandas Index of all the Components

        Get Pandas Index of a Component from the ComponentHandler by its
        positional identifier

        Args:
            index: int of the index of the component in the ComponentHandler
            with_meta: bool to include or not meta series in the return value

        Returns:
            Pandas Index of the names of the component
        """
        cols = []
        for i, c in enumerate(self.components):
            cols.extend(self.get_component_by_id(i, with_meta).to_list())
        return Index(cols)

    def copy(self, deep=True) -> 'ComponentHandler':
        """
        Copy function, deep by default

        Args:
            deep: bool if deep copy or not

        Returns:
            ComponentHandler
        """

        return deepcopy(self) if deep else copy(self)

    @staticmethod
    def __format_main_series(index: int, value: list):
        """ Format a main series name

        Args:
            index: int of the position of the main series
            value: list with the main series name

        Returns:
            list with the formatted str of the series
        """
        return [f"{index}_{v}" for v in value]

    @staticmethod
    def __format_meta_series(index, value):
        """ Format a meta series name(s)

        Args:
            index: int of the position of the meta series
            value: list with the meta series names

        Returns:
            list with the formatted str of the series
        """
        return [f"{index}-{v}" for v in value]
