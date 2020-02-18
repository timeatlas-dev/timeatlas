import json


class Metadata:
    """ Metadata associated to one or many TimeSeries

    This object is mainly used for the creation of a clean metadata file when
    exporting a TimeSeries or TimeSeriesDataset object

    Attributes:
        data: A List of object, usually TimeSeries or TimeSeriesDataset
        name: An optional name for the data represented by the Metadata
        path: An optional path where the data represented by the Metadata will
            be stored.
    """

    def __init__(self, name=None, path=None):
        self.data = []
        if name:
            self.name = name
        if path:
            self.path = path

    def to_json(self, pretty_print=False):
        """ Convert the current Metadata object into a JSON string

        Args:
            pretty_print: Boolean allowing for pretty printing

        Returns:
            A String containing the JSON

        """
        return json.dumps(self, default=lambda x: x.__dict__,
                          sort_keys=True, indent=2) \
            if pretty_print \
            else json.dumps(self, default=lambda x: x.__dict__,
                            sort_keys=True)

