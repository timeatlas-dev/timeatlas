from .anomalies import AnomalyABC
from .utils import get_operator
from .labeler import AnomalySetLabeler
from .config import AnomalyConfigParser

import pandas as pd
import numpy as np
from itertools import cycle
import warnings
from copy import copy
import math


class AnomalyGenerator():
    def __init__(self, data, conf_file, axis=0):

        # TODO: Write translation tool between this and Hindis tool
        # TODO: Maybe add the ABC to the tool of Hindi

        # TODO: Instead of "save" we return a TimeSeriesDataset-object according to TimeAtlas

        # assertions
        assert axis == 0 or axis == 1
        assert isinstance(data, pd.DataFrame)

        # setting the iterator for the dataframe -> 0 = rows; 1 = columns
        self.axis = axis
        if self.axis == 0:
            self.iterator = pd.DataFrame.iterrows
        elif self.axis == 1:
            raise NotImplementedError
            self.iterator = pd.DataFrame.iteritems

        # functions for anomaly
        self.ABC = AnomalyABC()

        # read the config file
        self.config = AnomalyConfigParser(config_file=conf_file)
        self.GLOBAL = self.config['GLOBAL']
        self.ANOMALIES = self.config['ANOMALIES']
        self.anomaly_functions = self.get_anomaly_function()
        self.percent = self.GLOBAL['percent']
        self.selection = self.GLOBAL['selection']
        self.percent = self.GLOBAL['percent']
        self.amount = self.GLOBAL['amount']
        self.outfile = self.GLOBAL['outfile']

        # create numpy-random.RandomState object
        self.seed = np.random.seed(self.GLOBAL['seed'])

        # adding a label column to the dataframe and creating the results anomaly labels
        self.labels = AnomalySetLabeler(data, self.axis)
        # we need to remove the label column/row to not interfere with the anomaly introduction
        # load data
        self.data = data.drop(labels=self.labels.label_columns, axis=np.abs(self.axis - 1))
        # from this point on it is unwise to use "data" instead of self.data -> removed labels

        # figure out the precision of the data
        self.precision = self.generation_precision(self.data)

    @staticmethod
    def precision_and_scale(x):
        '''

        Get the precision of a value

        Args:
            x: a (float) number

        Returns: the number of positions after the comma

        '''
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

    def generation_precision(self, df):
        '''

        Set the rounded average precision of the values inside a dataframe

        Args:
            df: dataframe

        Returns: integer of the precision

        '''
        precision_df = df.applymap(self.precision_and_scale)
        return int(round(precision_df.mean().mean()))

    def save(self):
        self.labels.finalize()
        self.data.to_csv(f'{self.outfile}_data.csv')
        self.labels.annotation.to_csv(f'{self.outfile}_labels.csv')

    def get_anomaly_function(self):
        '''s

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

    def chose_series_by_amount(self):

        if self.selection and self.percent and self.amount:
            warnings.warn(
                "'amount', 'selection' and 'percent' are all given a value. The priorities of them are: 1. amount \n, 2.selection\n 3.percent \n")

        return self.data.sample(n=self.amount, random_state=self.seed)

    def chose_series_by_selection(self):
        '''

        getting the given rows in the dataframe self.data as the anomaly_series

        Returns: anomaly series chosen by the user

        '''

        if self.selection and self.percent:
            warnings.warn("Both random and selection were set -> user selection supersedes random choice.")
        return self.data.iloc[self.selection]

    def chose_series_at_random(self):
        '''

        selects at random the given portion of the dataframe

        e.g. if self.percent = 0.01 it will select 1% of the dataframe as candidates for the anomalies.
        This 1% will then further be distributed to the given anomalies -> if num_anomalies = 2 -> 0.05% each

        Returns: rows selected for the introduction of anomalies

        '''

        return self.data.sample(frac=self.percent, replace=False, random_state=self.seed, axis=self.axis)

    def plot_anomaly_series(self):
        '''

        #TODO

        Plotting all series with inserted anomalies -> maybe with the original in the same plot

        Returns: Plots the anomalies

        '''
        raise NotImplementedError

    def add_data(self, new_data, name):

        if self.axis == 0:
            top, bottom = self.data.loc[:name].drop(labels=name, axis=self.axis), self.data.loc[name:].drop(labels=name,
                                                                                                            axis=self.axis)
            row_df = pd.DataFrame([new_data], index=[name])
            self.data = pd.concat([top, row_df, bottom], axis=self.axis, sort=False)
        elif self.axis == 1:
            raise NotImplementedError

    def generate(self):

        if self.amount:
            anomaly_series = self.chose_series_by_amount()
        elif self.selection:
            anomaly_series = self.chose_series_by_selection()
        else:
            anomaly_series = self.chose_series_at_random()

        zip_list_functions = self.create_zip_object(self.iterator(anomaly_series), self.anomaly_functions)

        for (name, data), (function, params) in zip_list_functions:
            operation_param = params['operation']
            function_params = copy(params)
            function_params.pop('operation')
            anomaly, coordinates = function(data, **function_params)

            # creating the new data to add
            operator = get_operator(mode=operation_param)
            new_data = operator(data, start=coordinates, values=anomaly)
            self.add_data(new_data, name)
            self.labels.create_operation_dict(coordinates, operation_param, name, function.__name__)

        # round all values to the same precision
        self.data = self.data.round(decimals=self.precision)

        if self.labels.original_labels is not None:
            self.data['original_labels'] = self.labels.original_labels

        if self.GLOBAL['save']:
            self.save()
