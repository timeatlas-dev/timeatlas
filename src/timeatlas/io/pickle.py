import pickle
from typing import Any


def read_pickle(path: str) -> Any:
    """
    Load an object from a pickle file

    Args:
        path: str of the path of the object on your file system

    Returns:
       The deserialized object
    """
    with open(path, 'rb') as f:
        return pickle.load(f)
