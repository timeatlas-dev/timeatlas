from timeatlas.abstract.abstract_base_generator import AbstractBaseGenerator
from timeatlas.time_series import TimeSeries
from timeatlas.time_series_dataset import TimeSeriesDataset

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
    def __init__(self, data: TimeSeriesDataset, conf_file):

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
    def precision_and_scale(x):
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
    def clean_parameters(values):
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

    def save(self):
        self.labels.finalize()
        self.data.to_text(f'./{self.outfile}_data')
        self.labels.annotation.to_csv(f'./{self.outfile}_data/{self.outfile}_labels.csv', index=False)

    def get_anomaly_function(self):
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

    def chose_amount(self):

        ind, data = self.data.random(n=self.amount, seed=self.seed, indices=True)
        return list(zip(ind, data))

    def chose_selection(self):
        ind, data = self.data.select(selection=self.selection, indices=True)
        return list(zip(ind, data))

    def chose_percentage(self):
        ind, data = self.data.percent(percent=self.percent, seed=self.seed, indices=True)
        return list(zip(ind, data))

    def plot_anomaly_series(self):
        """

        Plotting all series with inserted anomalies -> maybe with the original in the same plot

        Returns: Plots the anomalies

        """
        raise NotImplementedError

    def add_data(self, new_data, ind):

        self.data[ind].series = pd.Series(new_data)

    def generate(self):

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
            anomaly, coordinates = function(data, **function_params)
            # creating the new data to add
            operator = get_operator(mode=operation_param)
            new_data = operator(data, start=coordinates, values=anomaly)
            self.add_data(new_data, ind)
            self.labels.create_operation_dict(coordinates=coordinates,
                                              param=operation_param,
                                              function_name=function.__name__,
                                              name=ind,
                                              outfile=self.outfile)

            # round all values to the same precision
            ts.round(decimals=self.precision)

        if self.GLOBAL['save']:
            self.save()
