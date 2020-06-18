import numpy as np


def chunkify(tsd, seq_len):
    """

    Splits a TimeSeriesDataset into chunks of length seq_len

    Args:
        tsd: TimeSeriesDataset object
        seq_len: length of the subsequences to return

    Returns: numpy arrays with chunks of size seq_len

    """
    x, y = [], []
    for s in tsd:
        for i in range(len(s) - seq_len):
            x_i = s.series[i: i + seq_len]
            y_i = s.series[i + seq_len]

            x.append(x_i)
            y.append(y_i)

    return np.array(x), np.array(y)
