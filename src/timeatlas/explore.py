import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dt
from tqdm.notebook import tqdm

from src.timeatlas.data import Dataset


class Explore:

    def describe(self, dataset: Dataset):
        desc = []
        for obj_id, obj_df in tqdm(dataset.values.items()):
            obj_desc = pd.DataFrame(obj_df, columns=[obj_id]).describe()
            desc.append(obj_desc)
        return pd.concat(desc, axis=1)

    def compute_object_resolution(self, object_df: pd.DataFrame):
        """
        Compute the time between each measurement in a single object
        TODO Optimize the speed of this method

        :param object_df: DataFrame of the object
        :return: DataFrame of the differences between each measurements
        """
        res = []
        for k, v in enumerate(object_df.index[:-1]):
            delta = object_df.index[k+1] - object_df.index[k]
            res.append(delta)
        res_df = pd.DataFrame(res)

        try:
            return res_df[0].dt.total_seconds().values
        except Exception as e:
            print("type error: {}".format(str(e)))
            return None

    def compute_dataset_resolution(self, dataset: Dataset):
        """
        Compute the time between each measurement in all objects in a dataset
        TODO Optimize the speed of this method

        :param dataset: Dataset
        :return: Tuple[List, Dict] where the List is a summary of the time
        differences and the Dict stores all time differences for each object
        """
        res = {}
        desc = []
        for obj_id, obj_df in tqdm(dataset.values.items()):
            obj_res = self.compute_object_resolution(obj_df)
            try:
                res[obj_id] = obj_res
                desc.append(pd.DataFrame(obj_res, columns=[obj_id]).describe())
            except Exception as e:
                print("type error: {}".format(str(e)))
        desc = pd.concat(desc, axis=1)
        return res, desc


    def compute_time_frame(self, dataset:Dataset, plot=False):

        # Create a DataFrame with time frames
        time_frame = pd.DataFrame()
        for k, v in dataset.values.items():
            first = v.index[0]
            last = v.index[-1]
            time_frame[k] = [first, last]
        time_frame = time_frame.set_index(pd.Index(["first", "last"]))
        time_frame = time_frame.T

        # Plot the time frame if desired
        if plot:
            fig = plt.figure(figsize=(20, 10))
            ax = fig.add_subplot(111)
            ax = ax.xaxis_date()
            ax = plt.hlines(time_frame.index,
                            dt.date2num(time_frame["first"]),
                            dt.date2num(time_frame["last"]),
                            linewidth=1)
            ax = plt.title("First and last timestamp for every sensors")
            ax = plt.xlabel("Date")
            ax = plt.ylabel("Sensor ID")
            plt.plot()

        return time_frame

