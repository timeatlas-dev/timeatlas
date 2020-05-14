from typing import Any
import pandas as pd

from timeatlas.metadata import Metadata
from timeatlas.time_series import TimeSeries
from ._utils import check_directory_structure


def read_text(path: str) -> Any:
    """
    Create a TimeSeries object from a text file

    Args:
        path: a String representing the path to the file to read

    Returns: TimeSeries
    """
    # Check struct
    check_directory_structure(path)
    # Define dirs
    metadata_file = '{}/metadata.json'.format(path)
    # Create the Metadata object if existing
    my_metadata = Metadata()
    my_metadata.read(metadata_file)
    # As this method is related to one single TimeSeries, the metadata should only be related to one as well.
    assert len(my_metadata.data) == 1, "The quantity of time series in the dataset isn't equal to one."
    # Create the TimeSeries object with metadata
    ts_meta = my_metadata.data[0]
    ts_path = my_metadata.path.joinpath(ts_meta["path"])
    df = pd.read_csv(ts_path)
    df = df.set_index(pd.to_datetime(df["timestamp"]))
    df = df.drop(columns=["timestamp"])
    del ts_meta["path"]
    return TimeSeries(df["values"], ts_meta)
