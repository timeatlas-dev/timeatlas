from typing import NoReturn, Any
from pandas import Series
from timeatlas.metadata.metadata import Metadata
from timeatlas.abstract.abstract_io import AbstractIO
from timeatlas.utils import ensure_dir


class IO(AbstractIO):

    def read(self, path: str) -> Any:
        """
        - create a timeseries object, with or without metadata
        """


        pass

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
        full_path = path + file_name + ".csv"
        series.to_csv(full_path, header=True, index_label="timestamp")

    @staticmethod
    def __metadata_to_json_file(metadata: Metadata, path: str,
                                file_name: str = "metadata"):
        full_path = path + file_name + ".json"
        with open(full_path, "w") as file:
            file.write(metadata.to_json(pretty_print=True))

