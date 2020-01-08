from src.timeatlas.explore import Explore
from src.timeatlas.handle import Handle
from src.timeatlas.process import Process
from src.timeatlas.utils.smn_helper import SmnHelper


class Toolbox:

    def __init__(self):
        self.explore = Explore()
        self.handle = Handle()
        self.process = Process()
        self.smn_helper = SmnHelper()


toolbox = Toolbox()

