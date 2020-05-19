import pickle
from typing import NoReturn

from pandas import DataFrame, Series, to_datetime, to_numeric, read_csv

from timeatlas.metadata import Metadata
from timeatlas.archive import Archive
from timeatlas.utils import ensure_dir
from timeatlas.abstract import AbstractOutputText, AbstractOutputPickle


class IO(AbstractOutputText, AbstractOutputPickle):

    def to_text(self, path: str) -> NoReturn:

        data_dir_name = "data"
        index = str(0) #noqa

        # Create output directory
        output_dir = "{}".format(path)
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

            # Create the Package file
            package = Archive(name)
            package.data.append(self.metadata)
            self.__metadata_to_json_file(metadata, output_dir)

    def to_pickle(self, path: str, name: str) -> NoReturn:
        """
        Export a object in Pickle on your file system

        Args:
            data: The object to serialize
            path: str of the path to the target directory
            name: str of the name of the file
        """
        with open("{}/{}.pickle".format(path,name), 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)


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
    def __metadata_to_json_file(archive: Archive, path: str,
                                file_name: str = "metadata"):
        """
        Write a Metadata object into a JSON file

        Args:
            archive: the Metadata object
            path: the path where the Metadata will be saved
            file_name: the name given to the JSON file
        """
        full_path = path + file_name + ".json"
        with open(full_path, "w") as file:
            file.write(archive.to_json(pretty_print=True))

    @staticmethod
    def __csv_file_to_series(path: str):
        """
        Read a CSV file and put it into a Pandas Series

        Args:
            path: The path to read

        Returns: a Pandas Series object
        """

    @staticmethod
    def __values_to_dataframe(values):
        """
        Get the raw values output from BBData (JSON) into a dataframe for a given
        object and a time frame.

        :param object_id: Integer defining the BBData object
        :param from_timestamp: String defining a timestamp as "2018-01-01T00:00"
        :param to_timestamp: String defining a timestamp as "2018-02-01T00:00"
        :return: Pandas DataFrame containing the values
        """

        df = DataFrame(values)

        # Transform the timestamp column into datetime format
        try:
            df['timestamp'] = to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        except KeyError:
            print("Index conversion error")

        # Transform the values from object to a numerical value
        try:
            df = to_numeric(df.value)
        except ValueError:
            print("Value conversion error")

        return df

    @staticmethod
    def __load_object(object_id: int, dataset_path: str):
        """
        Private method to read a single object from a CSV file

        :param object_id: Integer defining the ID of the object
        :param dataset_path: String of the path to the dataset
        :return: Pandas DataFrame containing the values
        """
        file_path = "{}/{}.csv".format(dataset_path, object_id)
        df = read_csv(file_path, index_col="timestamp")
        if df.empty:
            raise Exception("The DataFrame is empty")
        else:
            # Converting the index as date
            df.index = to_datetime(df.index)
            return df