import os
from typing import NoReturn, Any
import pickle
from matplotlib.axes import Axes


def ensure_dir(file_path: str) -> NoReturn:
    """
    Make sure that a path to a directory exists, or creates its directory leafs

    Args:
        file_path: the path to the file/directory
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def to_pickle(obj: Any, path: str) -> NoReturn:
    """
    Export a object in Pickle on your file system

    Args:
        obj: Object to serialize
        path: str of the path to the target directory
    """
    ensure_dir(path)
    with open(path, 'wb') as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)


def add_metadata_to_plot(meta, ax: Axes):
    if meta is not None:
        if "unit" in meta:
            unit = meta["unit"]
            ax.set_ylabel("{} $[{}]$".format(unit.name, unit.symbol))
        if "sensor" in meta:
            sensor = meta["sensor"]
            ax.set_title("{}—{}".format(sensor.id, sensor.name))
    return ax
