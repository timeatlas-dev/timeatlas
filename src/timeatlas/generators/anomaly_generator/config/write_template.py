from typing import NoReturn

from configobj import ConfigObj
import inspect

from ..anomalies import AnomalyABC
from ..anomalies.utils import get_function_names


class AnomalyGeneratorTemplate(ConfigObj):
    def __init__(self, filename, seed: int = None, functions: list or str = "__all__",
            threshold: float = None, num_anomalies: int = None,
            anomaly_name: str = "ANOMALY"):
        super().__init__()

        # setting up the seed
        if seed is None:
            self.initial_comment.append("WARNING: No seed was set. This will make the results not reproducible")

        self.seed = seed

        # creating the anomaly object
        self.ABC = AnomalyABC(seed=self.seed)

        # check functions and number of anomalies
        if functions and num_anomalies:
            assert len(functions) == num_anomalies

        # setting up the list of anomalies and their functions
        if functions:
            # check if functions are defined as list, str or "__all__"
            if isinstance(functions, list):
                self.functions = functions
            elif isinstance(functions, str):
                if functions == '__all__':
                    # if all functions get them from the anomaly object
                    self.functions = get_function_names(self.ABC)
                else:
                    self.functions = [functions]

            self.num_anomalies = len(self.functions)

        # if only number of anomalies are given set up empty confic
        elif num_anomalies:
            self.num_anomalies = num_anomalies
        else:
            raise Exception("Either num_anomalies(int) or functions(list) has to be defined")

        # setup filename and comments
        self.filename = filename + '.ini'
        self.initial_comment = ["Automatically created config-file "]

        # some internal parameters by ConfigObj
        self.write_empty_values = True

        # set header name and anomaly_name
        self.header_name = 'GLOBAL'
        self.anomaly_name = anomaly_name + ' '

        self.threshold = threshold

        self.create_config()

    def create_config(self) -> NoReturn:
        """

        Creating the config file based on the input of the user

        Returns: NoReturn -> saves the file directly.

        """
        self[self.header_name] = {}
        self.inline_comments[
            self.header_name] = "!!One of the settings 'percent', 'selection' or 'amount' has to be set!!"
        self[self.header_name]['on_threshold'] = ''
        self[self.header_name]['seed'] = '' if self.seed is None else self.seed
        self[self.header_name]['percent'] = ''
        self[self.header_name]['selection'] = ''
        self[self.header_name]['amount'] = ''
        self[self.header_name]['outfile'] = ''
        self[self.header_name]['save'] = True

        self['ANOMALIES'] = {}
        self.inline_comments['ANOMALIES'] = "'operation' is either 'add', 'replace' or 'insert'"

        for i in range(0, self.num_anomalies, 1):
            n = i + 1
            self['ANOMALIES'][self.anomaly_name + str(n)] = {}
            self['ANOMALIES'][self.anomaly_name + str(n)]['PARAMETERS'] = {}
            if self.functions is not None:
                self['ANOMALIES'][self.anomaly_name + str(n)]['function'] = self.functions[i]
                params = self.anomaly_function_parameters(i)
                for param in params:
                    self['ANOMALIES'][self.anomaly_name + str(n)]['PARAMETERS'][param] = ''
            else:
                self['ANOMALIES'][self.anomaly_name + str(n)]['function'] = ''

    def anomaly_function_parameters(self, function_index: int) -> list:
        """

        Get the function parameters form the function so they can be written into the config.ini

        Args:
            function_index: index of the anomaly to the the paramters from

        Returns: List of parameters

        """
        try:
            params = inspect.getfullargspec(getattr(self.ABC, self.functions[function_index])).args
            # removing the two parameters self and data, that should not be given in the config file
            params.remove("self")
            params.remove("data")
            # adding the needed "operation" to the parameters
            params.insert(0, "operation")
            return params
        except Exception as e:
            fs = [bound_method[0] for bound_method in inspect.getmembers(self.ABC, predicate=(inspect.ismethod))]
            raise Exception("Declared function-name unknown: options: {}".format(str(fs))) from e
