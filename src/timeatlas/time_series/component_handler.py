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

    def __getitem__(self, item: Union[int, str, List[int], List[str]]):
        # handler[0]
        if isinstance(item, int):
            new_components = self.components[item]
        # handler["0_foo"]
        elif isinstance(item, str):
            new_components = self.get_component_by_name(item)

        elif isinstance(item, list):

            # handler[[0,3,5]]
            if all(isinstance(i, int) for i in item):
                new_components = [self.components[i] for i in item]

            # handler[["0_foo","1_bar"]]
            elif all(isinstance(i, str) for i in item):
                new_components = [self.get_component_by_name(i_n)
                                  for i_n in item]
            else:
                raise TypeError(f"ComponentHandler list indices must be int or "
                                f"str, not {type(item)}")
        else:
            raise TypeError(f"ComponentHandler indices must be int, str or list,"
                            f" not {type(item)}")

        return ComponentHandler(new_components)

    def __delitem__(self, key: Union[int, str]) -> NoReturn:
        """ Delete an item from the ComponentHandler

        Args:
            key: int or str of the item to delete
        """
        if isinstance(key, int):
            del self.components[key]
        elif isinstance(key, str):
            i = self.get_component_id_by_name(key)
            del self.components[i]

    def __len__(self) -> int:
        """ Get the number of item in the ComponentHandler

        Returns:
            int
        """
        return len(self.components)

    def __str__(self):
        """ get the str representation of a ComponentHandler

        Returns:
            str
        """
        return str(self.get_columns().to_list())

    def append(self, component: Component) -> NoReturn:
        """ Append a Component to the ComponentHandler

        Args:
            component: Component to append
        """
        self.components.append(component)

    def clear(self):
        """ Removes all Components from the ComponentHandler
        """
        self.components.clear()

    def get_component_id_by_name(self, name: str) -> int:
        """ Get a Component ID by its name

        Args:
            name: str of the name of the Component, including the ID (lol)
                e.g. "0_temperature"

        Returns:
            int
        """
        for i, c in enumerate(self.get_columns().to_list()):
            if name == c:
                return i
        # if no component are found throughout the for loop
        raise KeyError(f"Component with name '{name}' does not exist.")

    def get_component_by_name(self, name: str):
        """ Get a Component by its name

        Args:
            name: str of the name of the Component, including the ID
                e.g. "0_temperature"

        Returns:
            Component
        """
        for i, c in enumerate(self.components):
            component_name = self.__format_main_series(i, c.get_main())
            if name == component_name:
                return c
        raise KeyError(f"Component with name '{name}' does not exist.")

    def get_column_by_id(self, index: int) -> Index:
        """ Get a the name of a column by its Component ID

        Get Pandas Index of a Component from the ComponentHandler by its
        positional identifier

        Args:
            index: int of the index of the component in the ComponentHandler
            with_meta: bool to include or not meta series in the return value

        Returns:
            Pandas Index of the names of the component
        """
        c = self.components[index]
        cols = [self.__format_main_series(index, c.get_main())]
        return Index(cols)

    def get_column_by_name(self, name: str) -> Index:
        """ Get the name of a column by its Component name

        Args:
            name: str if the name of the component in the ComponentHandler
                  e.g: "0_temperature"

        Returns:
            Pandas Index of the names of the component
        """
        for i, c in enumerate(self.get_columns().to_list()):
            if name == c:
                return self.get_column_by_id(i)
        # if no component are found throughout the for loop
        raise KeyError(f"Component with name '{name}' does not exist.")

    def get_columns(self) -> Index:
        """ Get names of all the Components columns

        Get Pandas Index of a Component from the ComponentHandler by its
        positional identifier

        Args:
            index: int of the index of the component in the ComponentHandler

        Returns:
            Pandas Index of the names of the component
        """
        cols = []
        for i, c in enumerate(self.components):
            cols.extend(self.get_column_by_id(i).to_list())
        return Index(cols)

    def copy(self, deep=True) -> 'ComponentHandler':
        """ Copy function, deep by default

        Args:
            deep: bool if deep copy or not

        Returns:
            ComponentHandler
        """
        return deepcopy(self) if deep else copy(self)

    @staticmethod
    def __format_main_series(index: int, value: Union[str, list]):
        """ Format a main series name

        Args:
            index: int of the position of the main series
            value: list with the main series name

        Returns:
            list with the formatted str of the series
        """
        if isinstance(value, str):
            return f"{index}_{value}"
        elif isinstance(value, list):
            return [f"{index}_{v}" for v in value]
        else:
            TypeError(f"Type {value} isn't accepted")
