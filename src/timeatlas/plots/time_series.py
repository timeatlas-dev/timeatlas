from typing import TYPE_CHECKING, Any
from datetime import datetime

import numpy as np
import seaborn as sns
from matplotlib import (
    pyplot as plt,
    dates as mdates
)
from pandas.plotting import register_matplotlib_converters
import plotly.graph_objects as go
import plotly.io as pio

from timeatlas.config.colors import colors
from ._utils import add_metadata_to_plot
from timeatlas.config.constants import (
    TIME_SERIES_VALUES,
    TIME_SERIES_CI_LOWER,
    TIME_SERIES_CI_UPPER,
)

if TYPE_CHECKING:
    from timeatlas.time_series import TimeSeries

pio.templates.default = "plotly_white"
sns.set_style("whitegrid")
sns.set_context("notebook")


def line_plot(ts: 'TimeSeries', context: str = "paper", *args, **kwargs) -> Any:
    """Plot a TimeSeries

    This is a wrapper around Pandas.Series.plot() augmented if the
    TimeSeries to plot has associated Metadata.

    Args:
        ts: the TimeSeries to plot
        context: str defining on which medium the plot will be displayed
        *args: positional arguments for Pandas plot() method
        **kwargs: keyword arguments fot Pandas plot() method

    Returns:
        matplotlib.axes._subplots.AxesSubplot
    """
    if context == "paper":
        register_matplotlib_converters()

        if 'figsize' not in kwargs:
            kwargs['figsize'] = (18, 2)  # Default TimeSeries plot format

        if 'color' not in kwargs:
            kwargs['color'] = colors.blue_dark

        ax = ts.series.plot(*args, **kwargs)
        ax.set_xlabel("Date")
        ax.grid(True, c=colors.grey, ls='-', lw=1, alpha=0.2)

        # Add legend from metadata if existing
        if ts.metadata is not None:
            if "unit" in ts.metadata:
                unit = ts.metadata["unit"]
                ax.set_ylabel("{} $[{}]$".format(unit.name, unit.symbol))
            if "sensor" in ts.metadata:
                sensor = ts.metadata["sensor"]
                ax.set_title("{}—{}".format(sensor.id, sensor.name))

        return ax
    elif context == "notebook":
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ts.index,
            y=ts.values,
            name=ts.series[TIME_SERIES_VALUES].name
        ))
        fig.update_layout(showlegend=True)

        return fig
    else:
        raise ValueError("Context doesn't exit")


def prediction_plot(forecast: 'TimeSeries', observation: 'TimeSeries' = None) -> Any:
    """Plotting the prediction of a forecast

    Make a plot to display a time series with forecasted values and its
    confidence interval. If given, the observation time series can be added as
    an optional argument.

    Args:
        forecast: TimeSeries of the prediction
        observation: TimeSeries of the historical data

    Returns:
        matplotlib.axes._subplots.AxesSubplot
    """
    fig = plt.figure(figsize=(18, 4))
    ax = fig.add_subplot(111)
    ax.grid(True, which='major', c=colors.blue_dark, ls='-', lw=1, alpha=0.2)
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
    ax.plot(forecast.series.index, forecast.series[TIME_SERIES_VALUES].values,
            ls='--',
            c=colors.blue_dark,
            label="prediction")

    # If present, add confidence interval
    if TIME_SERIES_CI_LOWER in forecast.series.columns and \
            TIME_SERIES_CI_UPPER in forecast.series.columns:
        ax.fill_between(forecast.series.index,
                        forecast.series[TIME_SERIES_CI_LOWER].values,
                        forecast.series[TIME_SERIES_CI_UPPER].values,
                        color=colors.blue_light,
                        label="confidence interval")

    if observation is not None:
        ax.plot(observation.series.index, observation.series.values,
                ls='-',
                c=colors.blue_dark,
                label="observation")

    ax.legend()
    return ax


def status_plot(ts: 'TimeSeries', cmap: str = "autumn_r") -> Any:
    """plotting the status of a TimeSeries

    Plot a uni-dimensional imshow to mimic status plots like on
    https://githubstatus.com

    Args:
        ts: TimeSeries - the time series to plot
        cmap: String - the matplotlib colormap to use (default is "automn_r")

    Returns:
        matplotlib.axes._subplots.AxesSubplot
    """
    register_matplotlib_converters()

    fig, ax = plt.subplots(figsize=(18, 1))

    # Set x limits
    x_lims = [ts.boundaries()[0].to_pydatetime(), ts.boundaries()[1].to_pydatetime()]
    x_lims = mdates.date2num(x_lims)

    # Set y limits (for the sake of having something...)
    y_lims = [ts.min(), ts.max()]

    date_format = mdates.DateFormatter('%d/%m/%y %H:%M:%S')
    ax.xaxis.set_major_formatter(date_format)

    ax.set_yticks([])  # remove all yticks
    ax.xaxis_date()

    m = ax.imshow([ts.series],
                  extent=[x_lims[0], x_lims[1], y_lims[0], y_lims[1]],
                  aspect='auto',
                  cmap=cmap)

    ax = add_metadata_to_plot(ts.metadata, ax)

    fig.autofmt_xdate()
    plt.grid(b=True, which='both')
    plt.colorbar(m, aspect=5, pad=0.01)
    plt.show()
    return ax


def kde_plot(ts: 'TimeSeries') -> np.ndarray:
    """KDE plot

    Display a KDE plot through time with a line plot underneath

    Args:
        ts: TimeSeries - the time series to plot

    Returns:
        numpy.ndarray[matplotlib.axes._subplots.AxesSubplot]
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

    lower_plot.plot(timestamps, values, c=colors.blue_dark)
    lower_plot.xaxis.tick_top()
    lower_plot.grid()
    return axs
