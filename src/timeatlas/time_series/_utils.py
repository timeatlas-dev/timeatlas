from pandas import DataFrame, to_datetime, to_numeric, read_csv


class Utils:

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

