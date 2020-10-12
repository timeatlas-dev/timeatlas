from typing import NoReturn, Tuple, Any, Union, Optional, List, Callable, Dict

from timeatlas.abstract.abstract_base_generator import AbstractBaseGenerator
from timeatlas.time_series import TimeSeries
from timeatlas.time_series_dataset import TimeSeriesDataset
from timeatlas.config.constants import TIME_SERIES_VALUES

from .anomalies import AnomalyABC
from .utils import get_operator
from .labeler import AnomalySetLabeler
from .config import AnomalyConfigParser

import pandas as pd
import numpy as np
from itertools import cycle
from copy import copy
import math
from os import path


class AnomalyGenerator(AbstractBaseGenerator):
    """

    A generator that introcudes an anomaly into a given TimeSeriesDataset.

    The types and parameters are controlled with a .ini file,
    that can be created with "AnomalyGeneratorTemplate"

    """

    def __init__(self, data: TimeSeriesDataset, conf_file: str, save_as: str = 'text'):
        """

        Args:
            data: TimeSeriesDataset containing the data
            conf_file: config file created with AnomalyGeneratorTemplate
        """

        # Each generator set a label_suffix
        # Here: AGM -> Anomaly Generator Manual
        super().__init__()
        self.label_suffix = "AGM"
        self.save_as = save_as

        # assertions
        assert isinstance(data, TimeSeriesDataset)
        assert all(isinstance(x, TimeSeries) for x in
                   data), "One or more elements are not a TimeSeries-object"
        assert path.isfile(
            conf_file), f"No config file found under given path '{conf_file}'"

        # set data
        self.data = data

        # read the config file
        self.config = AnomalyConfigParser(config_file=conf_file)
        self.GLOBAL = self.config['GLOBAL']
        self.ANOMALIES = self.config['ANOMALIES']
        self.selection = self.GLOBAL['selection']
        self.percent = self.GLOBAL['percent']
        self.amount = self.GLOBAL['amount']
        self.outfile = self.GLOBAL['outfile']

        # create numpy-random.RandomState object
        self.seed = self.GLOBAL['seed']

        # functions for anomaly
        self.ABC = AnomalyABC(self.seed)
        self.anomaly_functions = self.get_anomaly_function()

        # adding a label column to the dataframe and creating the results anomaly labels
        self.labels = AnomalySetLabeler()

        # figure out the precision of the data
        self.precision = self.generation_precision()

    @staticmethod
    def precision_and_scale(x: float):
        """

        Get the precision of a value

        Args:
            x: a (float) number

        Returns: the number of positions after the comma

        """
        # 14 is the maximal number of digits python can handle (more is also unrealistic)
        max_digits = 14
        # if the number is NaN return nothing
        if math.isnan(x):
            return
        # figure out the magniture -> the numbers before the comma
        int_part = int(abs(x))
        magnitude = 1 if int_part == 0 else int(math.log10(int_part)) + 1
        if magnitude >= max_digits:
            return (magnitude, 0)
        # shift the number after the comma in front of the comma and figure out the amount
        frac_part = abs(x) - int_part
        multiplier = 10 ** (max_digits - magnitude)
        frac_digits = multiplier + int(multiplier * frac_part + 0.5)
        while frac_digits % 10 == 0:
            frac_digits /= 10
        scale = int(math.log10(frac_digits))
        return scale

    @staticmethod
    def clean_parameters(values) -> Dict:
        """
        Function to cleanup the parameters. If the parameter in the config-file are None, they are removed.
        Args:
            values: parameter values from he config files

        Returns: Dict of the paramters without the None

        """
        return {k: v for k, v in values['PARAMETERS'].items() if v is not None}

    @staticmethod
    def create_zip_object(data, anomaly_f):
        '''

        combines the two lists of the data, where the anomalies are added to and the anomaly-function

        if the function list is shorter it will cycle through them until all data has 1 anomaly

        if the data is shorter it will only assign one anomaly function

        Args:
            data: pd.Series of data
            anomaly_f: function of ABC.anomalies creating the anomaly

        Returns: zip-object

        '''

        # warnings.warn("Length of data > length of anomalies: Not all anomalies will be assigned.")

        zip_list = zip(data, cycle(anomaly_f))
        return zip_list

    def generation_precision(self):
        '''

        Set the rounded average precision of the values inside a dataframe

        Returns: rounded average number of digits after the comma

        '''

        precision_df = np.array(
            [self.precision_and_scale(x) for ts in self.data for x in
             ts.series.values])
        # This is more of a security. A correctly formated TimeSeries-object has no None elements
        precision_df = precision_df[precision_df != None]

        return int(round(precision_df.mean()))

    def save(self) -> NoReturn:
        """

        Saving the labels and the new TimeSeriesDataset to file.

        Returns: NoReturn

        """

        self.labels.finalize()

        if self.save_as == 'text':
            self.data.to_text(path=f'./{self.outfile}_data')
        elif self.save_as == 'pickle':
            self.data.to_pickle(path=f'./{self.outfile}_data.pkl')

        # This function is no longer needed, since we save the labels now in the TimeSeries
        # self.labels.annotation.to_csv(f'./{self.outfile}_data/{self.outfile}_labels.csv', index=False)

    def get_anomaly_function(self) -> List:
        '''

        Get all functions in the config file

        Returns: list of tuples with the functions as (function, parameters)

        '''
        functions = []
        for key, values in self.ANOMALIES.items():
            function = getattr(self.ABC, values['function'])
            # removing the keys with None
            parameters = self.clean_parameters(values)
            functions.append((function, parameters))
        return functions

    def chose_amount(self) -> List:
        """

        Chose the number of time windows based on a fixed amount given by the user in the config file:

        eg. amount = 10, will select 10 elements

        Returns: List of pair of indices and data

        """

        ind, data = self.data.random(n=self.amount, seed=self.seed, indices=True)
        return list(zip(ind, data))

    def chose_selection(self) -> List:
        """

        Chose the number of time windows based on a user selection given by the user in the config file:

        eg. selection = [0,1,5,9] will select the first, second, sixth and tenth element.

        Returns: List of pair of indices and data

        """
        ind, data = self.data.select(selection=self.selection, indices=True)
        return list(zip(ind, data))

    def chose_percentage(self) -> List:
        """

        Chose the number of time windows based on a user selection given by the user in the config file:

        e.g. percent = 0.2 will select 20% of the TimeSeriesDataset (min=0, max=1)

        Returns: List of pair of indices and data

        """
        ind, data = self.data.percent(percent=self.percent, seed=self.seed, indices=True)
        return list(zip(ind, data))

    def add_data(self, new_data: TimeSeries, index: int) -> NoReturn:
        """

        Replacing the old TimeSeries with the new TimeSeries containing the anomaly.

        Args:
            new_data: new TimeSeries that will replace the old one
            index: index of the TimeSeries to replace in the TimeSeriesDataset

        Returns: NoReturn

        """

        self.data[index].series[TIME_SERIES_VALUES].replace(to_replace=pd.Series(new_data))

    def add_labels(self, index, coordinates, function_name):
        """

        Create the labels that need to be added to the TimeSeries.
        Will create a new column for the labels and name them.

        Args:
            index: index of the TimeSeries in the TimeSeriesDataframe
            coordinates: start and end index of the anomaly in the TimeSeries
            function_name: label of the anomaly

        Returns:

        """
        labels = [None] * len(self.data[index].series)
        for coords in coordinates:
            start = coords[0]
            end = coords[1] + 1
            labels[start:end] = [function_name] * len(labels[start:end])
            self.data[index].series[f'label_{self.label_suffix}'] = labels
            self.data[index].label = function_name

    def generate(self) -> NoReturn:
        """
        raise NotImplementedError

        Main function to generate the anomalies.

        Returns: NoReturn

        """

        if self.amount:
            anomaly_series = self.chose_amount()
        elif self.selection:
            anomaly_series = self.chose_selection()
        else:
            anomaly_series = self.chose_percentage()

        zip_list_functions = self.create_zip_object(anomaly_series,
                                                    self.anomaly_functions)

        # TODO: This adds the anomalies at the start and not where they belong
        for (ind, ts), (function, params) in zip_list_functions:
            data = ts.series
            operation_param = params['operation']
            function_params = copy(params)
            function_params.pop('operation')
            # TODO: Here we make DataFrame -> Series. A more elegant solution is to be found
            anomaly, coordinates = function(data[TIME_SERIES_VALUES], **function_params)
            # creating the new data to add
            operator = get_operator(mode=operation_param)
            new_data = operator(data, start=coordinates, values=anomaly)
            # rounding the data to a precision typical for the given dataset
            new_data = new_data.round(decimals=self.precision)
            self.add_data(new_data=new_data, index=ind)
            self.labels.create_operation_dict(coordinates=coordinates,
                                              param=operation_param,
                                              function_name=function.__name__,
                                              name=ind,
                                              outfile=self.outfile)

            self.add_labels(index=ind,
                            coordinates=coordinates,
                            function_name=function.__name__)

        if self.GLOBAL['save']:
            self.save()
