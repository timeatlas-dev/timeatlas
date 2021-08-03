from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timeatlas.time_series import TimeSeriesDarts
from typing import Any

import pandas as pd
from numpy.random import normal
from numpy import arange, linspace, logspace, power, append, flip
from math import ceil

from timeatlas.abstract import AbstractBaseManipulator


class TimeShop(AbstractBaseManipulator):
    """

    A class for simple manipulations of a TimeSeries in a fluent interface.

    """

    def __init__(self, ts: 'TimeSeriesDarts'):
        ts._assert_univariate()
        self.time_series = ts

        # temporal saving of needed information
        self.clipboard = None

    # ==========================================================================
    # Private Functions
    # ==========================================================================

    def _set_index(self, start_time, end_time=None, n_values=None):

        timestamp = pd.Timestamp(start_time)
        if end_time:
            end_time = pd.Timestamp(end_time)
            index = pd.date_range(start=timestamp, end=end_time, freq=self.time_series.freq)
        elif n_values:
            index = pd.date_range(start=timestamp, periods=n_values, freq=self.time_series.freq)
        else:
            raise ValueError("Either 'end_time' or 'n_values' has to be set")
        return index

    # ==========================================================================
    # Decorators
    # ==========================================================================

    def _check_selector(func):
        """

        Checking before generating values that there are not already some saved.

        Returns: TimeShop

        """

        def wrapper(self, *args, **kwargs) -> TimeShop:

            force = kwargs.get('force')
            if force:
                func(self, *args, **kwargs)
                return self
            else:
                if self.clipboard is None:
                    func(self, *args, **kwargs)
                    return self
                else:
                    raise ValueError(f"There is a generated value waiting."
                                     f"\nCan be overwritten by setting TimeShop.generator_out = None"
                                     f"\n or by setting force=True")

        return wrapper

    def _check_manipulator(func):
        """

        Checking before generating values that there are not already some saved.

        Returns: TimeShop

        """

        def wrapper(self, *args, **kwargs) -> TimeShop:

            force = kwargs.get('force')
            if force:
                func(self, *args, **kwargs)
                return self
            else:
                if self.clipboard is not None:
                    func(self, *args, **kwargs)
                    return self
                else:
                    raise ValueError(f"There is a generated value waiting."
                                     f"\nCan be overwritten by setting TimeShop.generator_out = None"
                                     f"\n or by setting force=True")

        return wrapper

    def _check_operator(func):
        """

        Checking before inserting values that they have been created.

        Returns: TimeShop

        """

        def wrapper(self, *args, **kwargs) -> TimeShop:

            if self.clipboard is None:
                raise ValueError(f"There is no generated value.")
            else:
                func(self, *args, **kwargs)
                # removing the input
                self.clipboard = None
                return self

        return wrapper

    # ==========================================================================
    # Selectors
    # ==========================================================================

    @_check_selector
    def crop(self, start_time: str, end_time: str = None, n_values: int = None) -> TimeShop:
        """

            Remove n entries in the TimeSeries starting at timestamp (including).
            !WARNING!: Will immediately remove that part
        Args:
            start_time: start of the part to crop
            end_time: end of the part to crop (optional)
            n_values: number of values to crop after start (optional)

        Returns:

        """

        index = self._set_index(start_time=start_time, end_time=end_time, n_values=n_values)
        timestamp_before = index[0]
        timestamp_after = index[-1]

        if timestamp_before == self.time_series.start_time():
            _, self.time_series = self.time_series.split_after(split_point=timestamp_after)
        elif timestamp_after == self.time_series.end_time():
            self.time_series, _ = self.time_series.split_before(split_point=timestamp_before)
        else:
            tmp_before, tmp_after = self.time_series.split_before(split_point=timestamp_before)
            tmp_gap, tmp_after = tmp_after.split_after(split_point=timestamp_after)
            self.time_series = tmp_before.append(tmp_after.shift(-len(tmp_gap)))
        return self

    @_check_selector
    def copy(self, other: TimeSeriesDarts, start_time: str, end_time: str) -> Any:
        """

        Selecting a part of the original time series for further use (saved in self.clipboard)

        Args:
            start_time: start of the slice
            end_time: end of the slice

        Returns: NoReturn

        """

        timestamp_before = pd.Timestamp(start_time)
        timestamp_after = pd.Timestamp(end_time)

        if timestamp_before == other.start_time() and timestamp_after == other.end_time():
            self.clipboard = other
        elif timestamp_before == other.start_time():
            _, self.clipboard = other.split_after(split_point=timestamp_after)
        elif timestamp_after == other.end_time():
            self.clipboard, _ = other.split_before(split_point=timestamp_before)
        else:
            self.clipboard = other.slice(start_ts=timestamp_before, end_ts=timestamp_after)

    @_check_selector
    def threshold_search(self, threshold, operator):
        """

        TODO: Find all values below or above a threshold (even both) and apply one of the top function to them
        TODO: the problem is that it is not made to apply to multiply atm.

        Returns:

        """

        pass

    # ==========================================================================
    # Generators
    # ==========================================================================

    @_check_manipulator
    def flat(self, value: float = None) -> Any:
        """

        Args:
            value:

        Returns:

        """

        index = self.clipboard.time_index
        # if value is not set the value used is the first value in the clipboard TimeSeries
        if value is None:
            value = float(self.clipboard.values()[0])

        df = pd.DataFrame(data=[value] * len(index), index=index)
        self.clipboard = self.clipboard.from_dataframe(df=df,
                                                       freq=self.time_series.freq)

    @_check_manipulator
    def white_noise(self, sigma: float, mu: float = None):
        """



        Args:
            sigma:
            mu:

        Returns:

        """

        index = self.clipboard.time_index

        if mu:
            values = normal(loc=mu, scale=sigma, size=len(index))
        else:
            values = []
            for value in self.clipboard.values():
                values.append(float(normal(loc=value, scale=sigma, size=1)))

        df = pd.DataFrame(data=values, index=index)
        self.clipboard = self.clipboard.from_dataframe(df=df,
                                                       freq=self.time_series.freq)

    @_check_manipulator
    def trend(self, slope: float):
        """

        Args:
            slope:

        Returns:

        """

        index = self.clipboard.time_index
        values = slope * arange(0, len(index), 1)

        df = pd.DataFrame(data=values, index=index)
        self.clipboard = self.clipboard.from_dataframe(df=df,
                                                       freq=self.time_series.freq)

    @_check_manipulator
    def spike(self, spike_value: float, mode: str = 'lin', p: int = None):
        """

        Args:
            spike_value:
            mode:
            p:

        Returns:

        """

        length = len(self.clipboard)

        assert (length % 2) != 0, f"spread has to be an odd number; got {length}"

        # making sure that length = 1 or 3 is not the same
        # additionally this solves the issue with TimeSeries not allowing to be length 1
        length += 2
        assert length < len(self.time_series), f"created spike is longer that the original TimeSeries()"
        middle = ceil(length / 2)

        if mode == 'lin':
            first_part = linspace(0, spike_value, middle)
        elif mode == 'log':
            first_part = logspace(0, spike_value, middle)
        elif mode == 'exp':
            assert pow is not None, f"if mode='exp' the argument for the power p has to be set"
            start = power(0, 1 / float(p))
            stop = power(spike_value, 1 / float(p))
            first_part = power(linspace(start, stop, num=middle), p)
        else:
            raise ValueError("Unknown 'mode' given.")

        second_part = flip(first_part[:-1])

        values = append(first_part, second_part)[1:-1]
        index = self.clipboard.time_index
        df = pd.DataFrame(data=values, index=index)

        self.clipboard = self.clipboard.from_dataframe(df=df,
                                                       freq=self.time_series.freq)

    @_check_manipulator
    def shift(self, new_start: str):
        """

        shift clipboard timestamps

        Args:
            new_start:

        Returns:

        """

        values = self.clipboard.values()
        index = self._set_index(start_time=new_start, end_time=None, n_values=len(values))
        df = pd.DataFrame(data=values, index=index)

        self.clipboard = self.clipboard.from_dataframe(df=df,
                                                       freq=self.time_series.freq)

    # ==========================================================================
    # Operators
    # ==========================================================================

    @_check_operator
    def add(self) -> Any:
        """

        Adding given values on top of the existing values.

        Returns: TimeShop

        """
        # get the information
        timestamp_before = self.clipboard.start_time()
        timestamp_after = self.clipboard.end_time()

        if timestamp_before == self.time_series.start_time() and timestamp_after == self.time_series.end_time():
            self.clipboard = self.time_series + self.clipboard
        else:
            tmp = self.time_series.slice(start_ts=pd.Timestamp(timestamp_before), end_ts=timestamp_after)

            # add what is in the generated and the original at the timestamps
            self.clipboard = tmp + self.clipboard

        # replace it in time series
        self.replace()

    def multiply(self):
        """

        Multiplying given values on top of the existing values.

        Returns: TimeShop

        """
        timestamp_before = self.clipboard.start_time()
        timestamp_after = self.clipboard.end_time()

        if timestamp_before == self.time_series.start_time() and timestamp_after == self.time_series.end_time():
            self.clipboard = self.time_series * self.clipboard
        else:
            tmp = self.time_series.slice(start_ts=pd.Timestamp(timestamp_before), end_ts=timestamp_after)

            # multiply what is in the generated and the original at the timestamps
            self.clipboard = tmp * self.clipboard

        # replace it in time series
        self.replace()

    @_check_operator
    def replace(self) -> Any:
        """

        Replaces a part of the TimeSeries with the values stored in self.generator_out at the timestamp in self.timestamp.
        The replacement is including the given timestamp.

        Returns: TimeShop

        """
        timestamp_before = self.clipboard.start_time()
        timestamp_after = self.clipboard.end_time()

        if timestamp_before == self.time_series.start_time():
            # check if the complete series is to be replaced
            if timestamp_after == self.time_series.end_time():
                self.time_series = self.clipboard
            else:
                _, tmp_after = self.time_series.split_after(split_point=timestamp_after)
                self.time_series = self.clipboard.append(tmp_after)
        elif timestamp_after == self.time_series.end_time():
            tmp_before, _ = self.time_series.split_before(split_point=timestamp_before)
            self.time_series = tmp_before.append(self.clipboard)
        else:
            tmp_before, _ = self.time_series.split_before(split_point=timestamp_before)
            _, tmp_after = self.time_series.split_after(split_point=timestamp_after)
            self.time_series = tmp_before.append(self.clipboard).append(tmp_after)

    @_check_operator
    def insert(self) -> Any:
        """

        Inserts values stored in self.generator_out between the given timestamps t (self.timestamp) and t-1

        Returns: TimeShop

        """
        timestamp_before = self.clipboard.start_time()

        if timestamp_before == self.time_series.start_time():
            self.time_series = self.clipboard.append(self.time_series.shift(len(self.clipboard)))
        elif timestamp_before == self.time_series.end_time():
            self.time_series = self.time_series.append(self.clipboard)
        else:
            tmp_before, tmp_after = self.time_series.split_after(split_point=timestamp_before)
            self.time_series = tmp_before.append(self.clipboard).append(
                tmp_after.shift(len(self.clipboard)))

    # ==========================================================================
    # Utils
    # ==========================================================================

    def plot(self):
        self.time_series.plot(new_plot=True)

    def clean_clipboard(self):
        self.clipboard = None

    def extract(self) -> 'TimeSeriesDarts':
        """

        Returning the TimeSeries, that was worked on

        Returns: TimeSeries Object

        """
        return self.time_series
