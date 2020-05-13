import json
from pathlib import Path
from timeatlas.types import *
from .metadata_items import MetadataItems
from .metadata_encoder import MetadataEncoder


class Metadata:
    """ Metadata associated to one or many TimeSeries

    This object is mainly used for the creation of a clean metadata file when
    exporting a TimeSeries or TimeSeriesDataset object

    Attributes:
        data: A List of object, usually TimeSeries or TimeSeriesDataset
        name: An optional name for the data represented by the Metadata
        path: An optional path where the data represented by the Metadata is stored.
    """

    def __init__(self, name=None, path=None):

        self.data = []

        if name:
            self.name = name
        if path:
            self.path = path

    def from_json(self, path):
        """ Converts a JSON file into a Metadata object

        The known types in timeatlas.types will be serialized

        Args:
            path: String containing the path to the JSON file (./my-dir/metadata.json)

        Returns:
            A Metadata object
        """
        # Read JSON

        with open(path) as json_file:
            raw_json = json.load(json_file)

        # Set instance variable
        if 'data' in raw_json:
            for i in raw_json['data']:
                self.data.append(MetadataItems(i))

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
                          cls=MetadataEncoder,
                          sort_keys=True,
                          indent=2) \
            if pretty_print \
            else json.dumps(self,
                            default=lambda x: x.__dict__,
                            cls=MetadataEncoder,
                            sort_keys=True)
