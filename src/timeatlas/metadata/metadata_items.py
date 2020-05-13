from typing import Dict, Any
from timeatlas.types import *


class MetadataItems(dict):

    def __init__(self, items: Dict[int, Any] = None):
        super().__init__()
        # If the object is known, transform them!
        if items is not None:
            for k, v in items.items():
                if k == 'sensor':
                    self[k] = Sensor(v['sensor_id'], v['name'])
                elif k == 'unit':
                    self[k] = Unit(v['name'], v['symbol'], v['data_type'])
                else:
                    self[k] = v

    def to_json(self):
        return self
