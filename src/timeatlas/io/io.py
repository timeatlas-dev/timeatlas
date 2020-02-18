from timeatlas.abstract.abstract_io import AbstractIO
import pickle
from typing import Any, NoReturn


class IO(AbstractIO):

    def import_csv(self, path: str) -> Any:
        pass

    def export_csv(self, path: str) -> NoReturn:
        pass

    def import_pickle(self, path: str) -> Any:
        """
        Load an object from a pickle file

        Args:
            path: str of the path of the object on your file system

        Returns:
           The deserialized object
        """
        with open(path, 'rb') as f:
            return pickle.load(f)

    def export_pickle(self, data: Any, path: str) -> NoReturn:
        """
        Export a object in Pickle on your file system

        Args:
            data: The object to serialize
            path: str of the path to the target directory
        """
        with open(path, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
