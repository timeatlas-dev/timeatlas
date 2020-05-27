import warnings

import numpy as np
import pandas as pd


class AnomalySetLabeler:
    def __init__(self, data: pd.DataFrame, axis: int, ):
        self.axis = axis

        # check for existing labels
        self.label_columns = None
        self.original_labels = None
        self.infer_labels(data)

        # anomaly labels

        self.annotation = {}

    def create_operation_dict(self, coordinates, param, name, function_name):

        coordinates = sorted(coordinates, key=lambda x: x[0])
        # TODO: THis offset style is not good since it is reduncant witht he code in utils.insert
        offset = 0
        for coords in coordinates:
            try:
                next_key = int(max(self.annotation.keys())) + 1
            except:
                next_key = 0
            self.annotation[next_key] = {'start': coords[0] + offset,
                                         'end': coords[1] + offset,
                                         'function_name': function_name,
                                         'parameters': param,
                                         'dataframe_name': name
                                         }

            if param == 'insert':
                offset += coords[1] - coords[0]

    def finalize(self):
        self.annotation = pd.DataFrame.from_dict({i: self.annotation[i] for i in self.annotation.keys()},
                                                 orient='index')

    def infer_labels(self, data):
        '''

        Args:
            data:

        Returns:

        '''
        if self.axis == 0:
            rows = data.columns
            self.label_columns = [row for row in rows if 'label' in str(row).lower()]
        elif self.axis == 1:
            raise NotImplementedError
            columns = data.iterrows()
            self.label_columns = [col for col in columns if 'label' in str(col).lower()]

        if not self.label_columns:
            warnings.warn("No labels detected.")

        # TODO: Removed the adding of the labeling
        # self.labels[self.label_columns] = data[self.label_columns]
        self.original_labels = data[self.label_columns]

    def add_binary_label_column(self, data):
        if self.axis == 0:
            labels = np.zeros(data.shape[1])
            self.labels.loc['labels_binary'] = labels
        elif self.axis == 1:
            labels = np.zeros(data.shape[0])
            self.labels['label_binary'] = labels
        else:
            raise ValueError("The Parameter 'axis' as to be 1 or 0.")
