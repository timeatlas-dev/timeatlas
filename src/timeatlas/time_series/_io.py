import pickle
from typing import NoReturn
from pandas import Series

from timeatlas.config.constants import (
    TIME_SERIES_FILENAME,
    TIME_SERIES_FORMAT,
    METADATA_FILENAME,
    METADATA_FORMAT
)
from timeatlas.utils import ensure_dir
from timeatlas.abstract import AbstractOutputText, AbstractOutputPickle


class IO(AbstractOutputText, AbstractOutputPickle):

    def to_text(self, path: str) -> NoReturn:
        # Create the time series file
        file_path = "{}/{}.{}".format(path, TIME_SERIES_FILENAME, TIME_SERIES_FORMAT)
        ensure_dir(file_path)
        self.__series_to_csv(self.series, file_path)
        # Create the metadata file
        if self.metadata is not None:
            file_path = "{}/{}.{}".format(path, METADATA_FILENAME, METADATA_FORMAT)
            ensure_dir(file_path)
            self.metadata.to_json(pretty_print=True, path=file_path)

    def to_pickle(self, path: str) -> NoReturn:
        """
        Export a object in Pickle on your file system

        Args:
            path: str of the path to the target directory
        """
        ensure_dir(path)
        with open(path, 'wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def __series_to_csv(series: Series, path: str):
        """
        Read a Pandas Series and put it into a CSV file

        Args:
            series: The Series to write in CSV
            path: The path where the Series will be saved
        """
        series.to_csv(path, header=True, index_label="index")
