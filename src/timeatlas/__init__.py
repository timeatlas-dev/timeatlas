from src.timeatlas import core
from src.timeatlas.utils.smn_helper import SmnHelper


class Toolbox:

    def __init__(self):
        self.explore = core.Explore()
        self.handle = core.Handle()
        self.process = core.Process()
        self.smn_helper = SmnHelper()


toolbox = Toolbox()

