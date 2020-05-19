from typing import Dict, Any
from timeatlas.types import *


class Metadata(dict):

    def __init__(self, items: Dict[str, Any] = None):
        super().__init__()
        # If the object is known, transform them!
        if items is not None:
            self.add(items)

    def add(self, items: Dict[str, Any]):
        for k, v in items.items():
            if k == 'sensor':
                self[k] = v if isinstance(v, Sensor) else Sensor(v['sensor_id'], v['name'])
            elif k == 'unit':
                self[k] = v if isinstance(v, Unit) else Unit(v['name'], v['symbol'], v['data_type'])
            else:
                self[k] = v
