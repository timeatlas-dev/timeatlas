import pandas as pd
import numpy as np

from scipy.fft import fft, ifft

from .utils import start_indices


class AnomalyABC():
    """
        Basically just a collection of functions that can introduce anomalies into a data-series.
    """

    def __init__(self, seed: int or None = None):

        self.seed = seed
        np.random.seed(self.seed)

    def flatline(self, data: pd.Series, n: int = 1, minimum: int = 1, maximum: int or None = None):
        '''

        A sensor gets stuck at time t for a random length at the value of t-1

        Args:
            data: pandas Series of the data
            n: number of flatline events

        Returns: flatline events with coordinates where to integrate them

        '''

        if maximum is None:
            maximum = len(data)

        event_starts = start_indices(len(data), n)
        last_values = [data[e - 1] for e in event_starts]
        coordinates = []
        values = []
        for i, e in enumerate(event_starts):

            e_length = np.random.randint(minimum, maximum)
            e_max_length = data[e:e + e_length].shape[0]
            e_end = e + e_max_length
            coordinates.append([e, e_end])
            values.append(np.full((e_length), last_values[i])[:e_max_length])

        return np.array(values), coordinates

    def zeroing(self, data: pd.Series, n: int = 1, value: int = None, minimum: int = 1, maximum: int or None = None):
        '''

        A sensor stops transmitting values or falls on a default value for a random time period

        Args:
            data: pandas series with the data
            n: number of events
            value: default value (or None) the sensor falls on

        Returns: events with coordinates where to integrate them

        '''

        assert value is None or value == 0

        if maximum is None:
            maximum = len(data)

        event_starts = start_indices(len(data), n)
        coordinates = []
        values = []
        for i, e in enumerate(event_starts):

            e_length = np.random.randint(minimum, maximum)
            e_max_length = data[e:e + e_length].shape[0]
            e_end = e + e_max_length
            coordinates.append([e, e_end])
            values.append(np.full((e_length), value)[:e_max_length])

        return np.array(values), coordinates

    def outlier(self, data: pd.Series, n: int = 1, sigma: float = 0.1):
        '''

        Sudden 1 time index long spike in the time series

        Args:
            data: pandas series with the data
            n: number of events
            sigma: standard deviation of the normal distribution to draw the value to add from

        Returns: events with coordinates where to integrate them

        '''

        event_starts = start_indices(len(data), n)
        coordinates = []
        values = []
        for e in event_starts:
            offset = 1 if data[e] > 0 else -1
            values.append(data[e] + (np.random.normal(data[e], sigma, 1) * offset))
            coordinates.append([e, e])

        return np.array(values), coordinates

    def increase_noise(self, data: pd.Series, n: int = 1, sigma: float = 0.1,
            change_point_factor: float = 3.0, minimum: int = 1, maximum: int or None = None):
        '''

        # TODO: Make it possible so the noise can be decreased. (can we extract noise levels from input data)?

        A series of normal distributed values that are offset by a factor from the normal

        Args:
            data: pandas series
            n: number of events
            sigma: sigma of the normal distribution
            change_point_factor: factor * sigma for the offset of the anomaly

        Returns: events with coordinates where to integrate them

        '''

        if maximum is None:
            maximum = len(data)

        event_starts = start_indices(len(data), n)
        coordinates = []
        values = []

        for e in event_starts:
            offset = 1 if data[e] > 0 else -1
            start_value = data[e] + np.random.normal(data[e], change_point_factor * sigma,
                                                     1) * offset
            e_length = np.random.randint(minimum, maximum)
            e_max_length = data[e:e + e_length].shape[0]
            e_end = e + e_max_length

            coordinates.append([e, e_end])
            values.append(np.full((e_length), np.random.normal(start_value, sigma, e_length))[:e_max_length])

        return np.array(values), coordinates

    def change_point(self, data: pd.Series, n: int = 1, ending: bool = True,
            change_point_factor: int = 3, minimum: int = 1, maximum: int or None = None):
        '''

        A clear point t at which the data get offset to a different level

        Args:
            data: pandas time series
            n: number of events
            ending: if True the changepoint is reverted at a random time, else it goes until the end
            change_point_factor: sigma of the normal distributed change point

        Returns: events with coordinates where to integrate them

        '''

        if maximum is None:
            maximum = len(data)

        event_starts = start_indices(len(data), n)
        coordinates = []
        values = []
        end = len(data) - 1

        for e in event_starts:

            if ending:
                e_length = np.random.randint(minimum, maximum)
                end = e + e_length
            coordinates.append([e, end])
            # TODO: This give a line...introduction of noise?
            # TODO: Fixing this value might lead to issues
            offset = np.random.normal(0, change_point_factor, 1)
            # TODO: Fill this with normal distribution -> make this a user choice
            change = np.full((len(data[e:end])), offset)
            values.append(change)

        return np.array(values), coordinates

    def clipping(self, data: pd.Series, clip_value: float, mode: str = 'top'):
        """

        the time series gets clipped in the extreme values to a given user value

        Args:
            data: pandas time-series
            clip_value: value at which to cutoff.
            mode: 'top': only clip in positive, 'bottom' only in negative, 'both' do both

        Returns: events with coordinates where to integrate them

        """

        if mode == 'top':
            max_ind = np.where(data > clip_value)[0]
        elif mode == 'bottom':
            max_ind = np.where(data < clip_value)[0]
        else:
            invert_clip_value = clip_value * -1
            max_ind_top = np.where(data > clip_value)[0]
            max_ind_bot = np.where(data < invert_clip_value)[0]
            max_ind = np.unique(np.concatenate((max_ind_top, max_ind_bot), 0))

        start = np.random.choice(range(len(max_ind) + 1), 1, replace=False)[0]
        clip_ind = max_ind[start:]
        coordinates = []
        values = []
        for c in clip_ind:
            coordinates.append([c, c])
            values.append([clip_value])

        return np.array(values), coordinates

    def trend(self, data: pd.Series, slope: float, n: int = 1):
        '''

        the sensor as new trend in the data.

        Args:
            data: pandas series
            slope: level m of the trend according to y = m*x + b
            n: number of events

        Returns: events with coordinates where to integrate them

        '''

        event_starts = start_indices(len(data), n)
        values = []
        coordinates = []

        for start in event_starts:
            values.append(slope * np.arange(0, len(data[start:None]), 1))
            coordinates.append([start, len(data) - 1])

        return np.array(values), coordinates

    def electric_feedback(self, data: pd.Series, sampling_speed: int = 1, amp: float = 10, hrz: float = 50):
        '''

        Introducing an electric feedback into the TimeSeries using a Fast Fourier Transform.

        Args:
            data: TimeSeries object
            sampling_speed: frequency of the sampling (number of seconds between two indices)
            amp: amplitude of the feedback
            hrz: hertz of the frequency introduced

        Returns: TimeSeries object with the added frequency

        '''

        assert sampling_speed > 0

        # fft of the complete series
        N = len(data)
        #
        T = 1 / sampling_speed
        # we need to drop NaN so we can create the FFT
        yf = fft(data.dropna().to_numpy())
        xf = np.linspace(0.0, 1 / (2 * T), round(N / 2))

        # adding feedback at the given frequency (hrz)
        idx = (np.abs(xf - hrz)).argmin()
        yf[idx] += amp
        added_feedback_data = ifft(yf).real
        # extract a part of the new data and return it
        event_start = start_indices(len(yf), 1)

        # needs to return a list
        values = [added_feedback_data[event_start[0]:]]
        coordinates = [[event_start[0], len(data) - 1]]

        return np.array(values), coordinates

    def hard_knee(self, data: pd.Series, threshold: float, factor: float = 0.5, n: int = 1):
        '''

        Values above a threshold are squeezed by a factor (0 < factor <= 1)

        Args:
            data: pandas series
            threshold: values above this will be squeezed
            factor: squeeze factor
            n: number of events

        Returns: events with coordinates where to integrate them

        '''

        event_starts = start_indices(len(data), n)
        values = []
        coordinates = []

        for e in event_starts:
            for i, v in enumerate(data[e:None]):
                if np.abs(v) >= threshold:
                    values.append([data[i] * factor])
                    coordinates.append([i, i])

        return np.array(values), coordinates

    def max_smoothing(self, data: pd.Series, window_size: int, threshold: float, n: int = 1):
        '''

        values above a threshold will be smoothed by a moving average

        Args:
            data: pandas series
            window_size: size of the moving average window -> bigger window; more smooth
            threshold: value above the sensor vales will be "smoothend"
            n: number of events

        Returns: events with coordinates where to integrate them

        '''

        event_starts = start_indices(len(data), n)
        values = []
        coordinates = []

        for e in event_starts:
            window = np.zeros(window_size)
            for i, v in enumerate(data):
                window = np.append(window, [v])
                window = np.delete(window, 0)
                avg = np.mean(window)
                if i >= e and np.abs(v) >= threshold:
                    values.append([avg])
                    coordinates.append([i, i])

        return np.array(values), coordinates

    def ratio_compression(self, data: pd.Series, threshold: float, ratio: int = 4, n: int = 1):
        '''

        values above a threshold will be compressed by a ratio

        Args:
            data: pandas series
            threshold: values above will be compressed
            ratio: factor of the compression 1:x
            n: number of events

        Returns: events with coordinates where to integrate them

        '''

        event_starts = start_indices(len(data), n)
        values = []
        coordinates = []

        for e in event_starts:
            for i, v in enumerate(data):
                if i >= e and np.abs(v) >= threshold:
                    values.append([data[i] - (data[i] / ratio)])
                    coordinates.append([i, i])

        return np.array(values), coordinates

    def ripple(self):
        '''

        IDEA: There is a start point that makes the data-stream "unstable"...it will increase with time and then vanish

        Returns:

        '''
        pass
