from matplotlib.axes import Axes


def add_metadata_to_plot(meta, ax: Axes):
    if meta is not None:
        if "unit" in meta:
            unit = meta["unit"]
            ax.set_ylabel("{} $[{}]$".format(unit.name, unit.symbol))
        if "sensor" in meta:
            sensor = meta["sensor"]
            ax.set_title("{}—{}".format(sensor.id, sensor.name))
    return ax
