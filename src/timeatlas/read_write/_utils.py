import json
from os import path as os_path
import pandas as pd
from pandas import Series
from timeatlas.config.constants import *
from timeatlas import Metadata


def check_directory_structure(path: str):
    """
    Check if a directory given as dataset path has a structure corresponding
    to the timeatlas standard

    Args:
        path: path encoded in a string

    Returns:
        A Boolean representing the validity of the structure
    """

    is_data = os_path.exists("{}/{}.{}".format(path, TIME_SERIES_FILENAME, TIME_SERIES_EXT))
    is_meta = os_path.exists("{}/{}.{}".format(path, METADATA_FILENAME, METADATA_EXT))

    if is_data is True and is_meta is False:
        return "timeseries"
    elif is_data is True and is_meta is True:
        return "timeseries with metadata"
    else:
        return None


def csv_to_dataframe(path: str) -> Series:
    df = pd.read_csv(path)
    df = df.set_index(pd.to_datetime(df["index"]))
    df = df.drop(columns=["index"])
    return df["values"].to_frame()


def json_to_metadata(path: str) -> Metadata:
    with open(path) as json_file:
        meta = json.load(json_file)
    return Metadata(meta)

