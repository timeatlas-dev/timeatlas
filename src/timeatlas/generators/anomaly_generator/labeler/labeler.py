from typing import NoReturn
import pandas as pd


class AnomalySetLabeler:
    def __init__(self):
        """

        Class of the AnomalyGenerator Labels

        """
        self.annotation = {}

    def create_operation_dict(self, coordinates: list, param: str, function_name: str, name: str,
            outfile: str) -> NoReturn:
        """

        Create a dictionary of all the newly added anomalies.

        Args:
            coordinates: index of the start and end of the anomaly
            param: parameters used to create the anomaly
            function_name: name of the function that created the anomaly
            name: name of the dataframe used to insert the anomalies into. Filename
            outfile: Filename to save the labels to #deprecated

        Returns: NoReturn

        """
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
