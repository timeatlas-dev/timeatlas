import pandas as pd
from glob import glob
from warnings import warn

from typing import Any

from timeatlas.config.constants import *
from timeatlas.time_series import TimeSeries
from timeatlas.time_series_dataset import TimeSeriesDataset
from timeatlas.config.constants import METADATA_CLASS_LABEL
from ._utils import (
    check_directory_structure,
    csv_to_dataframe,
    json_to_metadata,
)


def read_text(path: str) -> Any:
    """Reading a singe TimeSeries from to_text()

    Create a TimeSeries object from a text file

    Args:
        path: a String representing the path to the file to read

    Returns:
       TimeSeries

    """

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
        ts = TimeSeries(series, metadata)
        if METADATA_CLASS_LABEL in ts.metadata:
            ts.class_label = ts.metadata[METADATA_CLASS_LABEL]
        return ts

    elif dir_type is None:
        raise IOError("The path doesn't' include any recognizable files")


def read_tsd(path: str) -> TimeSeriesDataset:
    """Read a to_text() to TSD

    Loading the data from the individual data "csv"-files. These files should be in the format created by
    the "to_text"-function, either by TimeSeries.to_text() or TimeSeriesDataset.to_text().

    Exp:

    data
    │    ├── TSD-folder
    │    │    ├── 0
    │    │    │   └── data.csv
    │    │    ├── 1
    │    │    │   └── data.csv
    │    │    ├── 2
    │    │    │   └── data.csv
    │    │    ├── 3
    │    │    │   └── data.csv
    │    │    ├── 4
    │    │    │   └── (if there is no data the folder will be skipped with a warning)
    │    │    └── 4
    │    │        └── data.csv

    Args:
        path: Path to the folder that contains the subfolder containing the individual data csv-files

    Returns:
        TimeSeriesDataset

    """
    ts_list = []

    folders = glob(f'{path}/*')
    for f in folders:
        try:
            ts_list.append(read_text(f))
        except IOError:
            warn(f'Folder "{f}" does not contain any recognizable files')

    return TimeSeriesDataset(ts_list)


def csv_to_tsd(path: str) -> 'TimeSeriesDataset':
    """Load csv-file as TimeSeresDataset

    Create a TimeSeriesDataset from a csv

    Args:
        path: the path to the csv file

    Returns: TimeSeriesDataset

    """
    tsd = []
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index)

    last_column_name = df.columns[-1]
    number_of_timeseries = int(last_column_name[0])

    for i in range(number_of_timeseries + 1):
        tmp = df.filter(regex=f'^{i}_')
        tmp.columns = [col.split(f"{i}_")[-1] for col in tmp.columns]
        tsd.append(TimeSeries(tmp))

    return TimeSeriesDataset(data=tsd)
