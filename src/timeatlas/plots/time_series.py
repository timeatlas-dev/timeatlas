from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters

from timeatlas.time_series import TimeSeries
from ._utils import add_metadata_to_plot


def prediction(ts: TimeSeries, pred: TimeSeries):
    """
    Make a plot to display a chunk of a TimeSeries and a prediction with its
    confidence interval

    Args:
        ts: TimeSeries of the historical data
        pred: TimeSeries of the prediction
    """
    fig = plt.figure(figsize=(18,4))
    ax = fig.add_subplot(111)
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)

    # Add contextual information to the plot
    if ts.metadata is not None:
        if "sensor" in ts.metadata:
            sensor = ts.metadata["sensor"]
            duration = str(pred.duration())
            sensor_str = "{}—{}".format(sensor.id, sensor.name)
            ax.set_title("Forecast of {} for {}".format(sensor_str, duration))
        else:
            ax.set_title("Forecast for {}".format(str(pred.duration())))

        if "unit" in ts.metadata:
            unit = ts.metadata["unit"]
            ax.set_ylabel("{} $[{}]$".format(unit.name, unit.symbol))

    ax.set_xlabel("Date")

    # Add the lines to the plot
    ax.plot(ts.series.index, ts.series.values, ls='-', c="0.3",
            label="Measurement")
    ax.plot(pred.series.index, pred.series["values"].values, ls='--', c='k',
            label="Prediction")
    ax.fill_between(pred.series.index, pred.series["ci_lower"].values,
                    pred.series["ci_upper"].values,
                    color='0.86',
                    label="Confidence Interval")
    ax.legend()


def status(ts: TimeSeries):
    """
    Plot a uni-dimensional imshow to mimic status plots like on
    https://githubstatus.com

    Args:
        ts: TimeSeries - the time series to plot
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

    m = ax.imshow([ts.series],
              extent=[x_lims[0], x_lims[1],  y_lims[0], y_lims[1]],
              aspect='auto',
              cmap="autumn_r")

    ax.xaxis_date()

    ax = add_metadata_to_plot(ts.metadata, ax)

    fig.autofmt_xdate()
    plt.grid(b=True, which='both')
    plt.colorbar(m, aspect=5, pad=0.01)
    plt.show()
