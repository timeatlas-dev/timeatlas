import warnings

import numpy as np
import pandas as pd


class AnomalySetLabeler:
    def __init__(self):
        # anomaly labels
        self.annotation = {}

    def create_operation_dict(self, coordinates, param,function_name, name, outfile):

        coordinates = sorted(coordinates, key=lambda x: x[0])
        # TODO: This offset style is not good since it is redundant with he code in utils.insert
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
                                         'dataframe_name': f'{outfile}_data/{name}/data.csv'
                                         }

            if param == 'insert':
                offset += coords[1] - coords[0]

    def finalize(self):
        self.annotation = pd.DataFrame.from_dict({i: self.annotation[i] for i in self.annotation.keys()},
                                                 orient='index')
