import os
from typing import NoReturn, Any
import pickle


def ensure_dir(file_path: str) -> NoReturn:
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
