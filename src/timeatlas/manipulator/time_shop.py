from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timeatlas.time_series import TimeSeriesDarts
from typing import Any, NoReturn

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
        self.generator_output = None

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
                return self

        return wrapper

    def _check_generator(func):
        """

        Checking before generating values that there are not already some saved.

        Returns: TimeShop

        """

        def wrapper(self, *arg, **kwargs) -> TimeShop:

            force = kwargs.get('force')
            if force:
                func(self, *arg, **kwargs)
                return self
            else:
                if self.generator_output is None:
                    func(self, *arg, **kwargs)
                    return self
                else:
                    raise ValueError(f"There is a generated value waiting."
                                     f"\nCan be overwritten by setting TimeShop.generator_out = None"
                                     f"\n or by setting force=True")

        return wrapper

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
        timestamp_before = self.generator_output.start_time()
        timestamp_after = self.generator_output.end_time()

        if timestamp_before == self.time_series.start_time() and timestamp_after == self.time_series.end_time():
            self.generator_output = self.time_series + self.generator_output
        else:
            tmp = self.time_series.slice(start_ts=pd.Timestamp(timestamp_before), end_ts=timestamp_after)

            # add what is in the generated and the original at the timestamps
            self.generator_output = tmp + self.generator_output

        # replace it in time series
        self.replace()

    def multiply(self):
        """

        Multiplying given values on top of the existing values.

        Returns: TimeShop

        """
        timestamp_before = self.generator_output.start_time()
        timestamp_after = self.generator_output.end_time()

        if timestamp_before == self.time_series.start_time() and timestamp_after == self.time_series.end_time():
            self.generator_output = self.time_series + self.generator_output
        else:
            tmp = self.time_series.slice(start_ts=pd.Timestamp(timestamp_before), end_ts=timestamp_after)

            # multiply what is in the generated and the original at the timestamps
            self.generator_output = tmp * self.generator_output

        # replace it in time series
        self.replace()

    @_check_operator
    def replace(self) -> Any:
        """

        Replaces a part of the TimeSeries with the values stored in self.generator_out at the timestamp in self.timestamp.
        The replacement is including the given timestamp.

        Returns: TimeShop

        """
        timestamp_before = self.generator_output.start_time()
        timestamp_after = self.generator_output.end_time()

        if timestamp_before == self.time_series.start_time():
            # check if the complete series is to be replaced
            if timestamp_after == self.time_series.end_time():
                self.time_series = self.generator_output
            else:
                _, tmp_after = self.time_series.split_after(split_point=timestamp_after)
                self.time_series = self.generator_output.append(tmp_after)
        elif timestamp_after == self.time_series.end_time():
            tmp_before, _ = self.time_series.split_before(split_point=timestamp_before)
            self.time_series = tmp_before.append(self.generator_output)
        else:
            tmp_before, _ = self.time_series.split_before(split_point=timestamp_before)
            _, tmp_after = self.time_series.split_after(split_point=timestamp_after)
            self.time_series = tmp_before.append(self.generator_output).append(tmp_after)

    @_check_operator
    def insert(self) -> Any:
        """

        Inserts values stored in self.generator_out between the given timestamps t (self.timestamp) and t-1

        Returns: TimeShop

        """
        timestamp_before = self.generator_output.start_time()

        if timestamp_before == self.time_series.start_time():
            self.time_series = self.generator_output.append(self.time_series.shift(len(self.generator_output)))
        elif timestamp_before == self.time_series.end_time():
            self.time_series = self.time_series.append(self.generator_output)
        else:
            tmp_before, tmp_after = self.time_series.split_after(split_point=timestamp_before)
            self.time_series = tmp_before.append(self.generator_output).append(
                tmp_after.shift(len(self.generator_output)))

    def crop(self, start_time: str, end_time: str = None, n_values: int = None) -> TimeShop:
        """

        Remove n entries in the TimeSeries starting at timestamp (including).

        Args:
            timestamp: timestamp of first element to remove
            n: number of elements removed

        Returns: TimeShop

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

    # ==========================================================================
    # Generators
    # ==========================================================================

    @_check_generator
    def flat(self, start_time: str, end_time: str = None, n_values: int = None, value: float = None, *args,
            **kwargs) -> Any:
        """

        Creating DataFrame of lengths n_values with either the values at timestamp or given by value.
        A given value will supersede the timestamp.

        Args:
            n_values: length of resulting array
            timestamp: timestamp of value to be taken and later usage
            value: value in the array

        Returns: TimeShop

        """
        timestamp = pd.Timestamp(start_time)

        # setting the index for the time series
        index = self._set_index(start_time=start_time, end_time=end_time, n_values=n_values)

        if value is None:
            value = float(self.time_series[timestamp].values())
            df = pd.DataFrame([value] * len(index), index=index)
            self.generator_output = self.time_series.from_dataframe(df=df,
                                                                    freq=self.time_series.freq)
        else:
            df = pd.DataFrame([value] * len(index), index=index)
            self.generator_output = self.time_series.from_dataframe(df=df,
                                                                    freq=self.time_series.freq)

    @_check_generator
    def white_noise(self, start_time: str, mu: float, sigma: float, end_time: str = None, n_values: int = None, *args,
            **kwargs):

        index = self._set_index(start_time=start_time, end_time=end_time, n_values=n_values)

        values = normal(loc=mu, scale=sigma, size=len(index))
        df = pd.DataFrame(values, index=index)

        self.generator_output = self.time_series.from_dataframe(df=df,
                                                                freq=self.time_series.freq)

    @_check_generator
    def trend(self, start_time: str, slope: float, end_time: str = None, n_values: int = None, *args, **kwargs):
        """

        Creating a trend TimeSeries()

        y = m*x + b

        Args:
            start_time: Timestamp at which the trend starts
            slope: steepness of the trend
            end_time: Timestamp at which the trend ends (mandatory if n_values not set)
            n_values: number of values (mandatory if end_time not set)
            *args:
            **kwargs:

        Returns:

        """

        index = self._set_index(start_time=start_time, end_time=end_time, n_values=n_values)

        values = slope * arange(0, len(index), 1)
        df = pd.DataFrame(values, index=index)

        self.generator_output = self.time_series.from_dataframe(df=df,
                                                                freq=self.time_series.freq)

    @_check_generator
    def copy(self, other: TimeSeriesDarts, start_copy: str, end_copy: str, insert_start: str):
        """

        Copy or a part of a TimeSeries and paste it into another TimeSeries()

        Args:
            other:
            start_time:
            end_time:

        Returns:

        """
        start_time = pd.Timestamp(start_copy)
        end_time = pd.Timestamp(end_copy)

        values = other.slice(start_ts=start_time, end_ts=end_time).values()
        index = self._set_index(start_time=insert_start, end_time=None, n_values=len(values))
        df = pd.DataFrame(values, index=index)

        self.generator_output = self.time_series.from_dataframe(df=df,
                                                                freq=self.time_series.freq)

    @_check_generator
    def spike(self, spike_time: str, spike_value: float, length: int, mode: str = 'lin', p: int = None):
        """

        Adding a spike with a given length

        Args:
            spike_time: time, where the spike is highest = spike_value
            spike_value: max value of the spike
            length: length of the spike
            mode: linear, logarithmic or exponential approach to the spike_value
            p: power if mode = exp

        Returns:

        """
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

        values = append(first_part, second_part)
        index = self._set_index(start_time=spike_time, n_values=len(values)).shift(-len(second_part))
        df = pd.DataFrame(values, index=index)

        self.generator_output = self.time_series.from_dataframe(df=df,
                                                                freq=self.time_series.freq)

    # ==========================================================================
    # Search
    # ==========================================================================

    def threshold_search(self, threshold, operator, function):
        """

        TODO: Find all values below or above a threshold (even both) and apply one of the top function to them
        TODO: the problem is that it is not made to apply to multiply atm.

        Returns:

        """
        assert hasattr(self.__class__, function), "chosen function does not exists"

    # ==========================================================================
    # Utils
    # ==========================================================================

    def plot(self):
        self.time_series.plot(new_plot=True)

    def clean_up(self):
        self.generator_output = None

    def extract(self) -> 'TimeSeriesDarts':
        """

        Returning the TimeSeries, that was worked on

        Returns: TimeSeries Object

        """
        return self.time_series
