import json
from pathlib import Path
from timeatlas.metadata import Metadata
from timeatlas.abstract import AbstractInput, AbstractOutputJson


class Archive(AbstractInput, AbstractOutputJson):
    """ Archive associated to one or many TimeSeries

    This class is used for the creation of a file when
    exporting a data structure like TimeSeries or TimeSeriesDataset object

    Attributes:
        data: A List of object, usually TimeSeries or TimeSeriesDataset
        name: An optional name for the data represented by the Metadata
        path: An optional path where the data represented by the Metadata is stored.
    """

    def __init__(self, name: str = None, path: str = None):
        self.data = []
        if name:
            self.name = name
        if path:
            self.path = path

    def read(self, path):
        """ Converts a JSON file into an Archive object

        The known types in timeatlas.types will be serialized

        Args:
            path: String containing the path to the JSON file (./my-dir/metadata.json)

        Returns:
            A Metadata object
        """
        # Read JSON
        with open(path) as json_file:
            raw_json = json.load(json_file)
        # Set instance variables
        if 'data' in raw_json:
            for i in raw_json['data']:
                self.data.append(Metadata(i))
        if 'name' in raw_json:
            self.name = raw_json['name']
        if 'path' in raw_json:
            self.path = Path(raw_json['path'])
        else:
            self.path = Path(path).parent

    def to_json(self, pretty_print=False):
        """ Convert the current Metadata object into a JSON string

        Args:
            pretty_print: Boolean allowing for pretty printing

        Returns:
            A String containing the JSON

        """
        return json.dumps(self,
                          default=lambda x: x.__dict__,
                          sort_keys=True,
                          indent=2) \
            if pretty_print \
            else json.dumps(self,
                            default=lambda x: x.__dict__,
                            sort_keys=True)
