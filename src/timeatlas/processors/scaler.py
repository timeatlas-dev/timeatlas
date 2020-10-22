from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timeatlas.time_series import TimeSeries


class Scaler:

    @staticmethod
    def minmax(ts: 'TimeSeries') -> 'TimeSeries':
        r"""Scale a TimeSeries within a [0,1] range (so called min max)

        .. math::
            x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}

        Args:
            ts: TimeSeries to scale

        Returns:
            TimeSeries
        """
        s = ts.series
        scaled_series = (s - s.min()) / (s.max() - s.min())
        return TimeSeries(scaled_series, ts.metadata)

    @staticmethod
    def zscore(ts: 'TimeSeries') -> 'TimeSeries':
        r"""Scale a TimeSeries values by removing the mean and scaling to unit
        variance

        .. math::
            z = \frac{x - \mu}{\sigma}

        where :
            - :math:`z` is the scaled value
            - :math:`x` is the value
            - :math:`\mu` is the mean of the time series
            - :math:`\sigma` is the standard deviation of the time series

        Args:
            ts: TimeSeries to scale

        Returns:
            TimeSeries
        """
        s = ts.series
        scaled_series = (s - s.mean()) / s.std()
        return TimeSeries(scaled_series, ts.metadata)
