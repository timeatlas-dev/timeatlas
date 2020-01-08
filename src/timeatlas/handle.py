import os
import pickle

from bbdata.endpoint import output
import pandas as pd
from tqdm.notebook import tqdm

from src.timeatlas.data import Dataset


class Handle:

    def __init__(self):
        self.objects_filename = "objects.csv"

    def import_from_pickle(self, path, name):
        """
        Load a dataset from a pickle file

        :param path: String of the path of the dataset on your file system
        :param name: String of the name of the Pickle file
        :return: the deserialized dataset object
        """
        with open(path + name + '.pkl', 'rb') as f:
            return pickle.load(f)

    def import_from_csv(self, path):
        """
        Load a BBData dataset from a CSV file.
        :param path: String of the path of the dataset on your file system
        """
        objects = pd.read_csv("{}/{}".format(path, self.objects_filename),
                              index_col=["id"])

        values = {}
        for obj_id in tqdm(objects.index):
            try:
                df = self.__load_object(obj_id, path)
                values[obj_id] = df
            except Exception as err:
                print("Object {} deleted : {}".format(obj_id, err))
                objects = objects.drop(obj_id)

        return Dataset(objects, values)

    def import_from_bbdata(self, object_ids, from_timestamp, to_timestamp,
                         path=None):
        """
        Load a dataset of object(s) with their metadata for a given time frame
        in CSV

        :param object_ids: Array of Integer defining the BBData object(s)
        :param from_timestamp: String defining a timestamp as "2018-01-01T00:00"
        :param to_timestamp: String defining a timestamp as "2018-02-01T00:00"
        :param path: String defining a path in your file system where the dataset
                    should be saved in CSV.
        :return: Pandas DataFrame containing the values
        """

        output.login()

        values = {}
        objects = []

        for obj_id in tqdm(object_ids):

            # Handle objects metadata
            meta = output.objects.get(obj_id)
            line = {
                "id": meta["id"],
                "name": meta["name"],
                "unit_symbol": meta["unit"]["symbol"],
                "unit_name": meta["unit"]["name"],
                "unit_type": meta["unit"]["type"]
            }
            objects.append(line)

            # Handle values
            raw = output.values.get(obj_id, from_timestamp, to_timestamp)[0]
            values[obj_id] = self.__values_to_dataframe(raw["values"])

        objects = pd.DataFrame(objects)
        output.logout()

        return Dataset(objects, values)

    @staticmethod
    def export_to_pickle(dataset: Dataset, path: str, name: str):
        """
        Export a Dataset instance in Pickle on your file system

        :param dataset: Dataset object
        :param path: String of the path to the target directory
        :param name: String of the name of the Dataset
        :return:
        """
        with open(path + name + '.pkl', 'wb') as f:
            pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def export_to_csv(dataset: Dataset, path: str, name: str):
        """
        Export a Dataset instance in CSV on your file system

        :param dataset: a Dataset
        :param path: String of the path to the target directory
        :param name: String of the name of the Dataset
        """

        if not os.path.exists(path + name):
            os.makedirs(path + name)

        # Export the values
        for obj_id in tqdm(dataset.objects.id):
            try:
                filename = path + str(obj_id) + ".csv"
                dataset.values[obj_id].to_csv(filename, header=True)
                print("{} : extraction done in {}".format(obj_id, filename))
            except AttributeError:
                print("{} : Object probably None".format(obj_id))
            print("{} : End".format(obj_id))

        # Export the objects
        filename = path + "objects.csv"
        dataset.objects.to_csv(filename, index=False)
        print("Objects metadatas saved in {}".format(filename))

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

        df = pd.DataFrame(values)

        # Transform the timestamp column into datetime format
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        except KeyError:
            print("Index conversion error")

        # Transform the values from object to a numerical value
        try:
            df = pd.to_numeric(df.value)
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
        df = pd.read_csv(file_path, index_col="timestamp")
        if df.empty:
            raise Exception("The DataFrame is empty")
        else:
            # Converting the index as date
            df.index = pd.to_datetime(df.index)
            return df
