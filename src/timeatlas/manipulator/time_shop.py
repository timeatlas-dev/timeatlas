from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timeatlas.time_series import TimeSeriesDarts
from typing import Any

import pandas as pd
import numpy as np
from math import ceil

from timeatlas.abstract import AbstractBaseManipulator
from .utils import operators


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
        """

        Creating a range of timestamps between two times.
        Either end_time or n_values has to be set. The frequency is given by self.time_series

        Args:
            start_time: start of timestamp range
            end_time: end of timestamp range
            n_values: number of values

        Returns: DateTimeRange

        """

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
            if force or self.clipboard is None:
                func(self, *args, **kwargs)
                if not isinstance(self.clipboard, list):
                    self.clipboard = [self.clipboard]
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
            if force or self.clipboard is not None:
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
    def random(self, length: int, seed: int = None) -> Any:
        """

        Selecting a part of self.time_series with a random start and a given length

        Args:
            length: length of the selected part (int)
            seed: setting the seed for the random number generator (int)

        Returns:

        """

        if seed:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        timestamp = pd.Timestamp(
            rng.choice(a=self.time_series.time_index[:len(self.time_series - (length - 1))], size=1)[0])

        self.clipboard = self.time_series.slice_n_points_after(start_ts=timestamp, n=length)

    @_check_selector
    def threshold_search(self, threshold, operator) -> Any:
        """

        Finding parts of self.time_series that are fulfilling the constraints given by threshold and operator.

        Operators possible are:

            "<": operator.lt,
             "<=": operator.le,
             "==": operator.eq,
             "!=": operator.ne,
             ">=": operator.ge,
             ">": operator.gt,

        given by the string representation

        Args:
            threshold: value overstepped for the search
            operator: type of overstepping

        Returns: No Return

        """

        if operator not in operators:
            raise ValueError(f"Unknown operator. Given, {operator}")

        # turning TimeSeries into DataFrame
        tmp = self.time_series.pd_dataframe()
        # selecting the values with the threshold
        tmp = tmp[operators[operator](tmp, threshold)]
        # splitting on the NaN
        tmp = np.split(tmp, np.where(np.isnan(tmp.values))[0])
        # remove NaN
        tmp = [ev[~np.isnan(ev.values)] for ev in tmp if not isinstance(ev, np.ndarray)]
        # remove empty DataFrames
        tmp = [ev for ev in tmp if not ev.empty]
        # turn back into TimeSeries
        self.clipboard = [self.time_series.from_dataframe(df) for df in tmp]

    # ==========================================================================
    # Generators
    # ==========================================================================

    @_check_manipulator
    def flat(self, value: float = None) -> Any:
        """

        Modifying self.clipboard into a flat time series

        Args:
            value: Value of the flat line. If not given the first value of the element in self.clipboard is taken.

        Returns: NoReturn

        """

        assert isinstance(self.clipboard, list)

        clipboard = []
        for clip in self.clipboard:
            index = clip.time_index
            # if value is not set the value used is the first value in the clipboard TimeSeries
            if value is None:
                value = float(clip.values()[0])

            df = pd.DataFrame(data=[value] * len(index), index=index)
            clipboard.append(clip.from_dataframe(df=df,
                                                 freq=self.time_series.freq))

        self.clipboard = clipboard

    @_check_manipulator
    def white_noise(self, sigma: float, mu: float = None):
        """

        Creating normal distributed values

        Args:
            sigma: Standard Deviation of the normal distribution
            mu: Mean of the normal distribution. If it is not given the mean is the value at the individual timestamp

        Returns: NoReturn

        """

        assert isinstance(self.clipboard, list)

        clipboard = []
        for clip in self.clipboard:
            index = clip.time_index

            if mu:
                values = np.random.normal(loc=mu, scale=sigma, size=len(index))
            else:
                values = []
                for value in clip.values():
                    values.append(float(np.random.normal(loc=value, scale=sigma, size=1)))

            df = pd.DataFrame(data=values, index=index)
            clipboard.append(clip.from_dataframe(df=df,
                                                 freq=self.time_series.freq))

        self.clipboard = clipboard

    @_check_manipulator
    def trend(self, slope: float, offset: float = 0) -> Any:
        """

        Creating a linear trend ( y = m*x + b)

        Args:
            slope: slope of the trend
            b: offset at x=0, default b=0

        Returns: NoReturn

        """

        assert isinstance(self.clipboard, list)

        clipboard = []
        for clip in self.clipboard:
            index = clip.time_index
            values = slope * np.arange(0, len(index), 1)

            df = pd.DataFrame(data=values, index=index)
            clipboard.append(clip.from_dataframe(df=df,
                                                 freq=self.time_series.freq))

        self.clipboard = clipboard

    @_check_manipulator
    def spike(self, spike_value: float, mode: str = 'lin', p: int = None):
        """

        Creating a spike. WWith the possibility of a different approach to the maximum.

        Possible modes:

        'lin': linear in- and decrease
        'exp': exponential in- and decrease (requires a set power p)
        'log': logarithmic in- and decrease

        Args:
            spike_value: Maximum value of the spike
            mode: how the approach to the maximum is done
            p: if mode is 'exp' the power of the increase

        Returns: NoReturn

        """

        assert isinstance(self.clipboard, list)
        clipboard = []
        for clip in self.clipboard:
            length = len(clip)

            assert (length % 2) != 0, f"spread has to be an odd number; got {length}"

            # making sure that length = 1 or 3 is not the same
            # additionally this solves the issue with TimeSeries not allowing to be length 1
            length += 2
            assert length < len(self.time_series), f"created spike is longer that the original TimeSeries()"
            middle = ceil(length / 2)

            if mode == 'lin':
                first_part = np.linspace(0, spike_value, middle)
            elif mode == 'log':
                first_part = np.logspace(0, spike_value, middle)
            elif mode == 'exp':
                assert pow is not None, f"if mode='exp' the argument for the power p has to be set"
                start = np.power(0, 1 / float(p))
                stop = np.power(spike_value, 1 / float(p))
                first_part = np.power(np.linspace(start, stop, num=middle), p)
            else:
                raise ValueError("Unknown 'mode' given.")

            second_part = np.flip(first_part[:-1])

            values = np.append(first_part, second_part)[1:-1]
            index = clip.time_index
            df = pd.DataFrame(data=values, index=index)

            clipboard.append(clip.from_dataframe(df=df,
                                                 freq=self.time_series.freq))

        self.clipboard = clipboard

    @_check_manipulator
    def shift(self, new_start: str):
        """

        Shifting the timestamps with the frequency given by self.time_series

        Args:
            new_start: Start point of the TimeSeries

        Returns: NoReturn

        """

        assert isinstance(self.clipboard, list)
        clipboard = []
        for clip in self.clipboard:

            values = clip.values()
            index = self._set_index(start_time=new_start, end_time=None, n_values=len(values))
            df = pd.DataFrame(data=values, index=index)

            clipboard.append(clip.from_dataframe(df=df,
                                                 freq=self.time_series.freq))

    # ==========================================================================
    # Operators
    # ==========================================================================

    @_check_operator
    def add(self) -> Any:
        """

        Adding given values on top of the existing values.

        Returns: TimeShop

        """

        assert isinstance(self.clipboard, list)
        clipboard = []
        for clip in self.clipboard:
            # get the information
            timestamp_before = clip.start_time()
            timestamp_after = clip.end_time()

            if timestamp_before == self.time_series.start_time() and timestamp_after == self.time_series.end_time():
                clipboard.append(self.time_series + clip)
            else:
                tmp = self.time_series.slice(start_ts=pd.Timestamp(timestamp_before), end_ts=timestamp_after)
                # add what is in the generated and the original at the timestamps
                clipboard.append(tmp + clip)

        self.clipboard = clipboard
        # replace it in time series
        self.replace()

    @_check_operator
    def multiply(self):
        """

        Multiplying given values on top of the existing values.

        Returns: TimeShop

        """

        assert isinstance(self.clipboard, list)
        clipboard = []
        for clip in self.clipboard:
            timestamp_before = clip.start_time()
            timestamp_after = clip.end_time()

            if timestamp_before == self.time_series.start_time() and timestamp_after == self.time_series.end_time():
                clipboard.append(self.time_series * clip)
            else:
                tmp = self.time_series.slice(start_ts=pd.Timestamp(timestamp_before), end_ts=timestamp_after)

                # multiply what is in the generated and the original at the timestamps
                clipboard.append(tmp * clip)

        self.clipboard = clipboard
        # replace it in time series
        self.replace()

    @_check_operator
    def replace(self) -> Any:
        """

        Replaces a part of the TimeSeries with the values stored in self.generator_out at the timestamp in self.timestamp.
        The replacement is including the given timestamp.

        Returns: TimeShop

        """

        assert isinstance(self.clipboard, list)

        for clip in self.clipboard:

            timestamp_before = clip.start_time()
            timestamp_after = clip.end_time()

            if timestamp_before == self.time_series.start_time():
                # check if the complete series is to be replaced
                if timestamp_after == self.time_series.end_time():
                    self.time_series = clip
                else:
                    _, tmp_after = self.time_series.split_after(split_point=timestamp_after)
                    self.time_series = clip.append(tmp_after)
            elif timestamp_after == self.time_series.end_time():
                tmp_before, _ = self.time_series.split_before(split_point=timestamp_before)
                self.time_series = tmp_before.append(clip)
            else:
                tmp_before, _ = self.time_series.split_before(split_point=timestamp_before)
                _, tmp_after = self.time_series.split_after(split_point=timestamp_after)
                self.time_series = tmp_before.append(clip).append(tmp_after)

    @_check_operator
    def insert(self) -> Any:
        """

        Inserts values stored in self.generator_out between the given timestamps t (self.timestamp) and t-1

        Returns: TimeShop

        """

        assert isinstance(self.clipboard, list)

        for clip in self.clipboard:

            timestamp_before = clip.start_time()

            if timestamp_before == self.time_series.start_time():
                self.time_series = clip.append(self.time_series.shift(len(clip)))
            elif timestamp_before == self.time_series.end_time():
                self.time_series = self.time_series.append(clip)
            else:
                tmp_before, tmp_after = self.time_series.split_after(split_point=timestamp_before)
                self.time_series = tmp_before.append(clip).append(
                    tmp_after.shift(len(clip)))

    # ==========================================================================
    # Utils
    # ==========================================================================

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

    def plot(self):
        """

        Plotting self.time_series

        Returns:

        """
        self.time_series.plot(new_plot=True)

    def clean_clipboard(self):
        self.clipboard = None

    def extract(self) -> 'TimeSeriesDarts':
        """

        Returning the TimeSeries, that was worked on

        Returns: TimeSeries Object

        """
        return self.time_series
