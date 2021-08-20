from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timeatlas.time_series import TimeSeriesDarts
from typing import Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import ceil

from timeatlas.abstract import AbstractBaseManipulator
from .utils import operators

OPERATOR_STRING = 'operator'
START_TIME_STRING = 'start_time'
END_TIME_STRING = 'end_time'


class TimeShop(AbstractBaseManipulator):
    """

    A class for simple manipulations of a TimeSeries in a fluent interface.

    """

    def __init__(self, ts: 'TimeSeriesDarts'):
        ts._assert_univariate()
        self.time_series = ts
        self.label_ts = None

        # temporal saving of needed information
        self.clipboard = None

        # anomalies added to the time_series
        self._anomalies = {}
        self.inserted_anomalies = None

    def __len__(self):
        return len(self.time_series)

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

    def _clipboard_gen(self):
        pass

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
                for clip in self.clipboard:
                    self._anomalies[len(self._anomalies) + 1] = {OPERATOR_STRING: func.__name__,
                                                                 START_TIME_STRING: clip.start_time(),
                                                                 END_TIME_STRING: clip.end_time()}
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
                # anoamly components are named the following was:
                # <type of anomaly>_<#of anomalies in total>
                num_anomalies = len(
                    self.inserted_anomalies.components) + 1 if self.inserted_anomalies is not None else 1
                for key, values in self._anomalies.items():
                    component_name = [f"{values[f'{OPERATOR_STRING}']}_{num_anomalies}"]
                    print(component_name)
                    anomaly_df = pd.DataFrame(data=[0] * len(self.time_series),
                                              index=self.time_series.time_index,
                                              columns=component_name)
                    index = self._set_index(start_time=values[START_TIME_STRING], end_time=values[END_TIME_STRING])
                    df = pd.DataFrame(data=[1] * len(index), index=index,
                                      columns=component_name)
                    anomaly_df.update(other=df, overwrite=True)
                    if self.inserted_anomalies is None:
                        self.inserted_anomalies = self.time_series.from_dataframe(anomaly_df)
                    else:
                        self.inserted_anomalies = self.inserted_anomalies.stack(self.time_series.from_dataframe(anomaly_df))
                # removing the input
                self._anomalies = {}
                self.clipboard = None
                return self

        return wrapper

    # ==========================================================================
    # Selectors
    # These functions select a part of self.time_series or a different TimeSeries()
    # that a user would like to work on and modify.
    # ==========================================================================

    @_check_selector
    def select(self, other: TimeSeriesDarts, start_time: str, end_time: str) -> Any:
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
            self.clipboard, _ = other.split_after(split_point=timestamp_after)
        elif timestamp_after == other.end_time():
            _, self.clipboard = other.split_before(split_point=timestamp_before)
        else:
            self.clipboard = other.slice(start_ts=timestamp_before, end_ts=timestamp_after)

    @_check_selector
    def select_random(self, length: int, seed: int = None) -> Any:
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
    # These functions generate new values within the selected part of a TimeSeries()
    # They usually replace the values that are within the selected part so some
    # precaution by the user is needed.
    # ==========================================================================

    @_check_manipulator
    def flatten(self, value: float = None) -> Any:
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

            df = pd.DataFrame(data=[value] * len(index), index=index, columns=self.time_series.components)
            clipboard.append(clip.from_dataframe(df=df,
                                                 freq=self.time_series.freq))

        self.clipboard = clipboard

    @_check_manipulator
    def create_white_noise(self, sigma: float, mu: float = None):
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

            df = pd.DataFrame(data=values, index=index, columns=self.time_series.components)
            clipboard.append(clip.from_dataframe(df=df,
                                                 freq=self.time_series.freq))

        self.clipboard = clipboard

    @_check_manipulator
    def create_trend(self, slope: float, offset: float = 0) -> Any:
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
            values = slope * np.arange(0, len(index), 1) + offset

            df = pd.DataFrame(data=values, index=index, columns=self.time_series.components)
            clipboard.append(clip.from_dataframe(df=df,
                                                 freq=self.time_series.freq))

        self.clipboard = clipboard

    @_check_manipulator
    def spiking(self, spike_value: float, mode: str = 'lin', p: int = None):
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
            df = pd.DataFrame(data=values, index=index, columns=self.time_series.components)

            clipboard.append(clip.from_dataframe(df=df,
                                                 freq=self.time_series.freq))

        self.clipboard = clipboard

    @_check_manipulator
    def time_shifting(self, new_start: str):
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
            df = pd.DataFrame(data=values, index=index, columns=self.time_series.components)

            clipboard.append(clip.from_dataframe(df=df,
                                                 freq=self.time_series.freq))

    # ==========================================================================
    # Operators
    # These functions offer different possibilities to put the selected and modified
    # part back into the original TimeSeries()
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

        time_series = self.time_series.pd_dataframe()

        for clip in self.clipboard:

            time_series.update(clip.pd_dataframe(), overwrite=True)

        self.time_series = self.time_series.from_dataframe(time_series)

    @_check_operator
    def insert(self) -> Any:
        """

        Inserts the clips in self.clipboard to the original time series and shifts the timestamps accordingly

        Returns: TimeShop

        """
        assert isinstance(self.clipboard, list)

        # Creating a DataFrame with the correct length.
        tmp_len = len(self.time_series) + sum([len(clip) for clip in self.clipboard])
        tmp_ind = pd.to_datetime(
            pd.date_range(start=self.time_series.start_time(), periods=tmp_len, freq=self.time_series.freq))
        # the dataframe contains an absurd high number so that it is obvious when something goes wrong
        time_series = pd.DataFrame([1000000000] * len(tmp_ind), index=tmp_ind, columns=self.time_series.components)

        # the timeseries is split into its parts with the insertions added between
        parts = []
        # keep track of the old time, where the last clip ended  so that we are not overlapping insertions and original time_series
        old_time = None
        # how much the preceding insertions have to be shifted in the time axis
        shift_int = 0
        for i, clip in enumerate(self.clipboard):
            # this is a special case so that it does not throw an error when insertion and original start at the same time
            if clip.start_time() == self.time_series.start_time():
                parts.append(clip)
                old_time = clip.end_time()
                shift_int += len(clip)
            else:
                # cutting the timeseries before the new insertion
                before, _ = self.time_series.split_before(clip.start_time())
                if old_time is not None:
                    # removing the part of the original before the preceding insertion
                    if old_time != self.time_series.start_time():
                        _, before = before.split_before(old_time)
                # adding and shifting the insertions to the correct timestamp
                parts.append(before.shift(shift_int))
                parts.append(clip.shift(shift_int))
                shift_int += len(clip)
                old_time = clip.end_time()

        # adding the part of the original time_series that comes after the last insertion
        try:
            _, after = self.time_series.split_before(old_time)
            parts.append(after.shift(shift_int))
        except ValueError:
            pass

        # updating the dataframe with the new values
        self.parts = parts
        for part in parts:
            time_series.update(part.pd_dataframe(), overwrite=True)

        # creating the new and overwriting the old time_series
        self.time_series = self.time_series.from_dataframe(time_series)

    # ==========================================================================
    # Utils
    # Some utilities that might be useful
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

    def clipboard_plot(self):
        """

        Plotting self.clipboard -> with red background

        Returns:

        """
        if len(self.clipboard) == 1:
            self.clipboard[0].plot()
        else:
            NotImplemented("Plotting of more than one intermediate TimeSeries in self.clipboard is not supported, yet.")

    def plot(self):
        """

        Plotting self.time_series -> with red background

        TODO: This nolonger works
        Returns:

        """
        self.time_series.plot(new_plot=True)
        for key, value in self._anomalies.items():
            s = value[START_TIME_STRING]
            e = value[END_TIME_STRING]
            plt.axvspan(s, e, facecolor='r', alpha=0.5)

    def clean_clipboard(self):
        self.clipboard = None

    def extract(self) -> 'TimeSeriesDarts':
        """

        Returning the TimeSeries, that was worked on

        Returns: TimeSeries Object

        """
        return self.time_series
