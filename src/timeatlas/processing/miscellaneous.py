from typing import List
from timeatlas import TimeSeries


def ceil(ts: TimeSeries, thresholds: List):

    def threshold(value: float, levels: List):
        if value < levels[0]:
            return 0
        elif levels[0] <= value < levels[1]:
            return 1
        elif value >= levels[1]:
            return 2

    return ts.apply(lambda x: threshold(x, thresholds))
