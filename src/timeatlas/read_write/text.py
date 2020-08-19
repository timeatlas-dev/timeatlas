import pandas as pd
from typing import Any

from timeatlas.config.constants import *
from timeatlas.time_series import TimeSeries
from timeatlas.time_series_dataset import TimeSeriesDataset
from ._utils import (
    check_directory_structure, csv_to_dataframe, json_to_metadata
)


def read_text(path: str) -> Any:
    """
    Create a TimeSeries object from a text file

    Args:
        path: a String representing the path to the file to read

    Returns:
       TimeSeries

    """

    #TODO: Add TimeSeriesDataset

    dir_type = check_directory_structure(path)

    if dir_type == "timeseries":
        data = "{}/{}.{}".format(path, TIME_SERIES_FILENAME, TIME_SERIES_EXT)
        series = csv_to_dataframe(data)
        return TimeSeries(series)

    elif dir_type == "timeseries with metadata":
        data = "{}/{}.{}".format(path, TIME_SERIES_FILENAME, TIME_SERIES_EXT)
        meta = "{}/{}.{}".format(path, METADATA_FILENAME, METADATA_EXT)
        series = csv_to_dataframe(data)
        metadata = json_to_metadata(meta)
        return TimeSeries(series, metadata)

    elif dir_type is None:
        raise IOError("The path doesn't' include any recognizable files")


def csv_to_tsd(path: str) -> 'TimeSeriesDataset':
    """

    Create a TimeSeriesDataset from a csv

    Args:
        path: the path to the csv file

    Returns: TimeSeriesDataset

    """
    tsd = []
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index)
    for i, d in df.iteritems():
        tsd.append(TimeSeries(d))

    return TimeSeriesDataset(data=tsd)
