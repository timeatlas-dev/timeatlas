import json
from typing import Dict, Any
from timeatlas.types import *
from timeatlas.utils import ensure_dir


class Metadata(dict):
    """
    Metadata of a TimeSeries-object
    """

    def __init__(self, items: Dict[str, Any] = None):
        super().__init__()
        # If the object is known, transform them!
        if items is not None:
            self.add(items)

    def add(self, items: Dict[str, Any]):
        """ Add an item to the metadata object

        If the type is recognized in TimeAtlas (see types module), this
        method creates a instance of this type and adds in the Metadata.

        Args:
            items: dict with the items to add

        Returns:
            Metadata (self)
        """
        for k, v in items.items():
            if k == 'sensor':
                self[k] = v if isinstance(v, Sensor) \
                    else Sensor(v['sensor_id'], v['name'])

            elif k == 'unit':
                self[k] = v if isinstance(v, Unit) \
                    else Unit(v['name'], v['symbol'], v['data_type'])

            elif k == 'coords':
                self[k] = v if isinstance(v, Coords) \
                    else Coords(v['lat'], v['long'])

            else:
                self[k] = v
        return self

    def to_json(self, pretty_print=False, path: str = None):
        """ Convert the current Metadata object into a JSON string

        Args:
            pretty_print: Boolean allowing for pretty printing
            path: String of the JSON file to write

        Returns:
            A String containing the JSON

        """
        my_json = json.dumps(self,
                             default=lambda x: x.__dict__,
                             sort_keys=True,
                             indent=2) \
            if pretty_print \
            else json.dumps(self,
                            default=lambda x: x.__dict__,
                            sort_keys=True)

        if path is not None:
            ensure_dir(path)
            with open(path, 'w', encoding='utf-8') as file:
                file.write(my_json)

        return my_json
