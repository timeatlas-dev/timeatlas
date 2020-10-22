from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from timeatlas.time_series import TimeSeries


def ceil(ts: 'TimeSeries', thresholds: List) -> 'TimeSeries':
    """Checking on a threshold

    Ceil (TODO change name. proposal: stepper, threshold_overtaking, ...) is a
    function providing a time series informing the user if a threshold has been
    trespassed with natural numbers. For instance:

    0 = no trespassing,
    1 = first level trespassed,
    2 = second level trespassed, etc...

    Args:
        ts: TimeSeries on which the data will be analyzed
        thresholds: List of threshold

    Returns:TimeSeries of threshold trespassing

    """

    def threshold(value: float, levels: List) -> TimeSeries:
        """Checking if a value above thrshold

        Args:
            value: threshold
            levels: levels of stepping over

        Returns:
            TimeSeries

        """
        for k, v in enumerate(levels):
            if value > v:
                if k + 1 == len(levels):
                    # meaning there's no other levels...
                    return k + 1
                else:
                    continue
            else:
                return k

    return ts.apply(lambda x: threshold(x, thresholds))
