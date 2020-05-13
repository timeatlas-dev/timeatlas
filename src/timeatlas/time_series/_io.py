from typing import NoReturn, Any
from pandas import Series
import pandas as pd
from timeatlas.metadata.metadata import Metadata
from timeatlas.abstract.abstract_io import AbstractIO
from timeatlas.utils import ensure_dir


class IO(AbstractIO):

    def read(self, path: str) -> Any:
        """
        - create a time series object, with or without metadata
        """
        # Check struct
        self.__check_directory_structure(path)

        # Define dirs
        metadata_file = '{}/metadata.json'.format(path)
        data_dir = '{}/data'.format(path)

        # Create the Metadata object if existing
        my_metadata = Metadata()
        my_metadata.from_json(metadata_file)

        # As this method is related to one single TimeSeries, the metadata should only be related to one as well.
        assert len(my_metadata.data) == 1, "The quantity of time series in the dataset isn't equal to one."

        # Create the TimeSeries object with metadata
        ts_meta = my_metadata.data[0]
        ts_path = my_metadata.path.joinpath(ts_meta["path"])
        df = pd.read_csv(ts_path)
        df = df.set_index(pd.to_datetime(df["timestamp"]))
        df = df.drop(columns=["timestamp"])
        del ts_meta["path"]

        # TODO find a way to return a TimeSeries
        return df["values"], ts_meta

    def write(self, path: str, name: str) -> NoReturn:
        data_dir_name = "data"
        index = str(0) #noqa

        # Create output directory
        output_dir = "{}/{}/".format(path, name)
        ensure_dir(output_dir)

        # Create data directory
        data_dir = "{}/{}/".format(output_dir, data_dir_name)
        ensure_dir(data_dir)

        # Create the time series file
        self.__series_to_csv_file(self.series, data_dir, index)

        if self.metadata is not None:
            # Add path to the TimeSeries metadata
            ts_path = "./{}/{}.csv".format(data_dir_name, index)
            self.metadata["path"] = ts_path

            # Create the metadata file
            metadata = Metadata(name)
            metadata.data.append(self.metadata)
            self.__metadata_to_json_file(metadata, output_dir)


    @staticmethod
    def __series_to_csv_file(series: Series, path: str, file_name: str):
        """
        Read a Pandas Series and put it into a CSV file with a given name

        Args:
            series: The Series to write in CSV
            path: The path where the Series will be saved
            file_name: The name given to the CSV file
        """
        full_path = path + file_name + ".csv"
        series.to_csv(full_path, header=True, index_label="timestamp")

    @staticmethod
    def __metadata_to_json_file(metadata: Metadata, path: str,
                                file_name: str = "metadata"):
        """
        Write a Metadata object into a JSON file

        Args:
            metadata: the Metadata object
            path: the path where the Metadata will be saved
            file_name: the name given to the JSON file
        """
        full_path = path + file_name + ".json"
        with open(full_path, "w") as file:
            file.write(metadata.to_json(pretty_print=True))

    @staticmethod
    def __csv_file_to_series(path: str):
        """
        Read a CSV file and put it into a Pandas Series

        Args:
            path: The path to read

        Returns: a Pandas Series object
        """

    @staticmethod
    def __check_directory_structure(path: str):
        """
        Check if a directory given as dataset path has a structure corresponding
        to the timeatlas standard

        Args:
            path: path encoded in a string

        Returns:
            A Boolean representing the validity of the structure
        """
        pass