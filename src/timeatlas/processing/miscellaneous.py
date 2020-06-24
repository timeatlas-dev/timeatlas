from typing import List, Union
from timeatlas import TimeSeries


def ceil(ts: TimeSeries, thresholds: List):

    def threshold(value: float, levels: List):
        for k, v in enumerate(levels[:-1]):
            print(k)

            lower_bound_index = k
            upper_bound_index = k + 1
            level_i = levels[lower_bound_index]
            level_j = levels[upper_bound_index]

            if value < level_i:
                return lower_bound_index
            elif level_i <= value < level_j:
                return upper_bound_index
            elif value >= level_j:
                continue

    return ts.apply(lambda x: threshold(x, thresholds))
