from typing import NoReturn, Any

from timeatlas.abstract.abstract_io import AbstractIO


class IO(AbstractIO):

    def import_pickle(self, path: str) -> Any:
        pass

    def export_pickle(self, path: str) -> NoReturn:
        pass

    def import_csv(self, path: str) -> Any:
        """
        Load a TimeSeries from a CSV file.
        :param path: String of the path of the TimeSeries on your file system
        """
        pass
        # objects = pd.read_csv("{}/{}".format(path, self.objects_filename),
        #                       index_col=["id"])
        #
        # values = {}
        # for obj_id in tqdm(objects.index):
        #     try:
        #         df = self.__load_object(obj_id, path)
        #         values[obj_id] = df
        #     except Exception as err:
        #         print("Object {} deleted : {}".format(obj_id, err))
        #         objects = objects.drop(obj_id)
        #
        # return Dataset(objects, values)

    def export_csv(self, path: str) -> NoReturn:
        """
        Export a TimeSeries instance in CSV on your file system

        :param time_series_dataset: a TimeSeriesDataset
        :param path: String of the path to the target directory
        :param name: String of the name of the TimeSeries
        """
        pass
        # if not os.path.exists(path + name):
        #     os.makedirs(path + name)
        #
        # # Export the values
        # for obj_id in tqdm(dataset.objects.id):
        #     try:
        #         filename = path + str(obj_id) + ".csv"
        #         dataset.values[obj_id].to_csv(filename, header=True)
        #         print("{} : extraction done in {}".format(obj_id, filename))
        #     except AttributeError:
        #         print("{} : Object probably None".format(obj_id))
        #     print("{} : End".format(obj_id))
        #
        # # Export the objects
        # filename = path + "objects.csv"
        # dataset.objects.to_csv(filename, index=False)
        # print("Objects metadatas saved in {}".format(filename))
