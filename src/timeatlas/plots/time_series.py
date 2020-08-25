from typing import TYPE_CHECKING
from datetime import datetime

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters

if TYPE_CHECKING:
    from timeatlas.time_series import TimeSeries
from ._utils import add_metadata_to_plot


def line(ts: 'TimeSeries', *args, **kwargs):
    """
    Plot a TimeSeries

    This is a wrapper around Pandas.Series.plot() augmented if the
    TimeSeries to plot has associated Metadata.

    Args:
        ts: the TimeSeries to plot
        *args: positional arguments for Pandas plot() method
        **kwargs: keyword arguments fot Pandas plot() method
    """
    register_matplotlib_converters()

    if 'figsize' not in kwargs:
        kwargs['figsize'] = (18, 2)  # Default TimeSeries plot format

    if 'color' not in kwargs:
        kwargs['color'] = "k"

    ax = ts.series.plot(*args, **kwargs)
    ax.set_xlabel("Date")
    ax.grid(True, c='gray', ls='-', lw=1, alpha=0.2)

    # Add legend from metadata if existing
    if ts.metadata is not None:
        if "unit" in ts.metadata:
            unit = ts.metadata["unit"]
            ax.set_ylabel("{} $[{}]$".format(unit.name, unit.symbol))
        if "sensor" in ts.metadata:
            sensor = ts.metadata["sensor"]
            ax.set_title("{}—{}".format(sensor.id, sensor.name))


def prediction(forecast: 'TimeSeries', observation: 'TimeSeries' = None):
    """
    Make a plot to display a time series with forecasted values and its
    confidence interval. If given, the observation time series can be added as
    an optional argument.

    Args:
        forecast: TimeSeries of the prediction
        observation: TimeSeries of the historical data
    """
    fig = plt.figure(figsize=(18, 4))
    ax = fig.add_subplot(111)
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    ax.set_xlabel("Date")

    # Add contextual information to the plot
    if forecast.metadata is not None:
        if "sensor" in forecast.metadata:
            sensor = forecast.metadata["sensor"]
            duration = str(forecast.duration())
            sensor_str = "{}—{}".format(sensor.id, sensor.name)
            ax.set_title("Forecast of {} for {}".format(sensor_str, duration))
        else:
            ax.set_title("Forecast for {}".format(str(forecast.duration())))

        if "unit" in forecast.metadata:
            unit = forecast.metadata["unit"]
            ax.set_ylabel("{} $[{}]$".format(unit.name, unit.symbol))

    # Add the lines to the plot
    ax.plot(forecast.series.index, forecast.series["values"].values,
            ls='--',
            c='k',
            label="prediction")
    ax.fill_between(forecast.series.index, forecast.series["ci_lower"].values,
                    forecast.series["ci_upper"].values,
                    color='0.86',
                    label="confidence interval")

    if observation is not None:
        ax.plot(observation.series.index, observation.series.values,
                ls='-',
                c="0.3",
                label="observation")

    ax.legend()


def status(ts: 'TimeSeries', cmap: str = "autumn_r"):
    """
    Plot a uni-dimensional imshow to mimic status plots like on
    https://githubstatus.com

    Args:
        ts: TimeSeries - the time series to plot
        cmap: String - the matplotlib colormap to use
    """
    register_matplotlib_converters()

    fig, ax = plt.subplots(figsize=(18,1))

    # Set x limits
    x_lims = [ts.boundaries()[0].to_pydatetime(),ts.boundaries()[1].to_pydatetime()]
    x_lims = mdates.date2num(x_lims)

    # Set y limits (for the sake of having something...)
    y_lims = [ts.min().values[0], ts.max().values[0]]

    date_format = mdates.DateFormatter('%d/%m/%y %H:%M:%S')
    ax.xaxis.set_major_formatter(date_format)

    ax.set_yticks([])  # remove all yticks
    ax.xaxis_date()

    m = ax.imshow([ts.series],
              extent=[x_lims[0], x_lims[1],  y_lims[0], y_lims[1]],
              aspect='auto',
              cmap=cmap)

    ax = add_metadata_to_plot(ts.metadata, ax)

    fig.autofmt_xdate()
    plt.grid(b=True, which='both')
    plt.colorbar(m, aspect=5, pad=0.01)
    plt.show()


def kde(ts: 'TimeSeries'):
    """
    Display a KDE plot through time with a line plot underneath

    Args:
        ts: TimeSeries - the time series to plot
    """
    timestamps = ts.series.index.astype(np.int64) // 10 ** 9
    values = ts.series["values"].values

    fig, axs = plt.subplots(2, 1,
                            sharex='col',
                            figsize=(16, 5),
                            gridspec_kw={'height_ratios': [4, 1]})

    upper_plot = axs[0]
    lower_plot = axs[1]

    sns.kdeplot(timestamps, values,
                shade=True,
                cmap="YlOrRd",
                shade_lowest=False,
                ax=upper_plot)

    x_ticks = upper_plot.get_xticks()
    x_labels = [datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M') for t in x_ticks]

    upper_plot = add_metadata_to_plot(ts.metadata, upper_plot)
    upper_plot.set_xticklabels(x_labels)
    upper_plot.xaxis.tick_bottom()

    lower_plot.plot(timestamps, values, c='k')
    lower_plot.xaxis.tick_top()
    lower_plot.grid()
