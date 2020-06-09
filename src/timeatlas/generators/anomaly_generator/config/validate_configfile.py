from configobj import ConfigObj
from validate import Validator
from os import path


class AnomalyConfigParser(ConfigObj):
    def __init__(self, config_file):

        super().__init__(config_file, configspec=f'{path.dirname(__file__)}/.configspec.ini')

        self.val = AnomalyConfigValidator()

        # Per default its False
        self.valid_config = False
        self.offending = []

        self._validate()

    def _validate(self):

        assert 'percent' in self['GLOBAL'] or 'selection' in self['GLOBAL'] or 'amount' in self['GLOBAL'], \
            "Either a percentage (percent), a row selection (selection) or a number of rows (amount) as to be set."

        _validator_output = self.validate(self.val)
        if _validator_output is True:
            self.valid_config = True
        else:
            self.get_offending(_validator_output)

        if self.offending:
            raise ValueError("Error in {}".format(self.offending))

    def get_offending(self, validation: dict):
        for keys, values in validation.items():
            if isinstance(values, dict):
                self.get_offending(values)
            elif isinstance(values, bool):
                if not values:
                    self.offending.append((keys, values))


class AnomalyConfigValidator(Validator):
    def __init__(self):
        super().__init__()
