import numpy as np


def chunkify(arr, seq_len):
    x, y = [], []
    for s in arr:
        for i in range(len(s) - seq_len):
            x_i = s[i: i + seq_len]
            y_i = s[i + seq_len]

            x.append(x_i)
            y.append(y_i)

    return np.array(x), np.array(y)


