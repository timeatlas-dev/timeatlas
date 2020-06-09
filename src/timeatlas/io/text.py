from typing import Any
from timeatlas.config.constants import *
from timeatlas.time_series import TimeSeries
from ._utils import check_directory_structure, csv_to_series, json_to_metadata


def read_text(path: str) -> Any:
    """
    Create a TimeSeries object from a text file
    TODO: Add TimeSeriesDataset

    Args:
        path: a String representing the path to the file to read

    Returns:
       TimeSeries

    """
    dir_type = check_directory_structure(path)

    if dir_type == "timeseries":
        data = "{}/{}.{}".format(path, TIME_SERIES_FILENAME, TIME_SERIES_EXT)
        series = csv_to_series(data)
        return TimeSeries(series)

    elif dir_type == "timeseries with metadata":
        data = "{}/{}.{}".format(path, TIME_SERIES_FILENAME, TIME_SERIES_EXT)
        meta = "{}/{}.{}".format(path, METADATA_FILENAME, METADATA_EXT)
        series = csv_to_series(data)
        metadata = json_to_metadata(meta)
        return TimeSeries(series, metadata)

    elif dir_type is None:
        raise IOError("The path doesn't' include any recognizable files")
