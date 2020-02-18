import json


class Metadata:

    def __init__(self, name=None, path=None):
        self.data = []
        if name:
            self.name = name
        if path:
            self.path = path

    def to_json(self, pretty_print=False):
        return json.dumps(self, default=lambda x: x.__dict__,
                          sort_keys=True, indent=2) \
            if pretty_print \
            else json.dumps(self, default=lambda x: x.__dict__,
                            sort_keys=True)

