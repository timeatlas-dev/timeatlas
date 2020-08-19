from typing import List


def ceil(ts: 'TimeSeries', thresholds: List):
    """
    Ceil (TODO change name. proposal: stepper, threshold_overtaking, ...) is a
    function providing a time series informing the user if a threshold has been
    trespassed with natural numbers. For instance:

    0 = no trespassing,
    1 = first level trespassed,
    2 = second level trespassed, etc...

    :param ts: TimeSeries on which the data will be analyzed
    :param thresholds: List of threshold
    :return: TimeSeries of threshold trespassing
    """

    def threshold(value: float, levels: List):
        for k, v in enumerate(levels):
            if value > v:
                if k+1 == len(levels):
                    # meaning there's no other levels...
                    return k+1
                else:
                    continue
            else:
                return k

    return ts.apply(lambda x: threshold(x, thresholds))
