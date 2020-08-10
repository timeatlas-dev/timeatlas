from matplotlib import pyplot as plt


def prediction(ts, pred):
    """
    Make a plot to display a chunk of a TimeSeries and a prediction with its
    confidence interval

    :param ts: TimeSeries of the historical data
    :param pred: TimeSeries of the prediction
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



