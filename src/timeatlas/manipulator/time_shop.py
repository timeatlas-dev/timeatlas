from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timeatlas.time_series import TimeSeriesDarts
from typing import Any, NoReturn

import pandas as pd

from timeatlas.abstract import AbstractBaseManipulator


class TimeShop(AbstractBaseManipulator):
    """

    A class for simple manipulations of a TimeSeries in a fluent interface.

    """

    def __init__(self, ts: 'TimeSeriesDarts'):
        ts._assert_univariate()
        self.time_series = ts

        # temporal saving of needed information
        self.generator_output = None
        self.timestamp = None

    # ==========================================================================
    # Decorators
    # ==========================================================================

    def _check_operator(func):
        """

        Checking before inserting values that they have been created.

        Returns: TimeShop

        """

        def wrapper(self, *arg, **kwargs) -> TimeShop:
            if self.generator_output is None:
                raise ValueError(f"There is no generated value.")
            else:
                func(self, *arg, **kwargs)
                # removing the input
                self.generator_output = None
                self.timestamp = None
                return self

        return wrapper

    def _check_generator(func):
        """

        Checking before generating values that there are not already some saved.

        Returns: TimeShop

        """

        def wrapper(self, *arg, **kwargs) -> TimeShop:
            if self.generator_output is None:
                func(self, *arg, **kwargs)
                return self
            else:
                raise ValueError(f"There is a generated value waiting.")

        return wrapper

    # ==========================================================================
    # Operators
    # ==========================================================================

    @_check_operator
    def add(self) -> NoReturn:
        """

        Adding given values on top of the existing values.

        Returns: TimeShop

        """
        pass

    @_check_operator
    def replace(self) -> Any:
        """

        Replaces a part of the TimeSeries with the values stored in self.generator_out at the timestamp in self.timestamp.
        The replacement is including the given timestamp.

        Returns: TimeShop

        """
        tmp = self.time_series.slice_n_points_after(start_ts=pd.Timestamp(self.timestamp), n=len(self.generator_output))
        self.generator_output.index = tmp.time_index()
        self.time_series = self.time_series.update(index=tmp.time_index(), values=self.generator_output)

    @_check_operator
    def insert(self) -> Any:
        """

        Inserts values stored in self.generator_out between the given timestamps t (self.timestamp) and t-1

        Returns: TimeShop

        """
        tmp = self.time_series.slice_n_points_after(start_ts=pd.Timestamp(self.timestamp), n=len(self.generator_output))
        self.generator_output.index = tmp.time_index()
        tmp_before, tmp_after = self.time_series.split_before(ts=pd.Timestamp(self.timestamp))
        tmp_after = tmp_after.shift(len(tmp))
        self.time_series = tmp_before.append_values(self.generator_output).append(tmp_after)

    def crop(self, timestamp: str, n: int) -> TimeShop:
        """

        Remove n entries in the TimeSeries starting at timestamp (including).

        Args:
            timestamp: timestamp of first element to remove
            n: number of elements removed

        Returns: TimeShop

        """
        timestamp_before = pd.Timestamp(timestamp)
        timestamp_after = self.time_series.slice_n_points_after(start_ts=pd.Timestamp(timestamp_before), n=n).end_time()
        tmp_before, tmp_after = self.time_series.split_before(ts=timestamp_before)
        tmp_gap, tmp_after = tmp_after.split_after(ts=timestamp_after)
        self.time_series = tmp_before.append(tmp_after.shift(-len(tmp_gap)))
        return self

    # ==========================================================================
    # Generators
    # ==========================================================================

    @_check_generator
    def flat(self, n_values: int, timestamp: str, value: float = None) -> Any:
        """

        Creating DataFrame of lengths n_values with either the values at timestamp or given by value.
        A given value will supersede the timestamp.

        Args:
            n_values: length of resulting array
            timestamp: timestamp of value to be taken and later usage
            value: value in the array

        Returns: TimeShop

        """
        self.timestamp = pd.Timestamp(timestamp)
        if value is None:
            value = float(self.time_series[self.timestamp].values)
            self.generator_output = pd.DataFrame([value] * n_values)
        else:
            self.generator_output = pd.DataFrame([value] * n_values)

    @_check_generator
    def noise(self):
        pass

    @_check_generator
    def trend(self):
        pass

    # ==========================================================================
    # Utils
    # ==========================================================================

    def plot(self):
        self.time_series.plot()

    def extract(self) -> 'TimeSeriesDarts':
        """

        Returning the TimeSeries, that was worked on

        Returns: TimeSeries Object

        """
        return self.time_series
