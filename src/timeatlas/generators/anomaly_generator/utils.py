import pandas as pd


def get_operator(mode):
    '''

    Args:
        mode: name of the mode

    Returns: mode to operate with the array

    '''

    # TODO: This could be solved more elegant with the inspect-library. For now this is good enough
    operations = {'insert': insert,
                  'replace': replace,
                  'add': add}

    return operations[mode]


def add(data, start, values):
    """

    Args:
        data: DataFrame of where we enter the new data -> only the row with "values" will be used
        start: list of start coordinates
        values: list of values to add

    Returns: Pandas Series with added anomaly

    """

    data = data['values'].values

    for e, v in zip(start, values):
        data[e[0]:e[0] + len(v)] = data[e[0]:e[0] + len(v)] + v

    return pd.Series(data)


def replace(data, start, values):
    '''

    e[0] is the starting coordinate
    e[1] is the ending coordinate

    Args:
        data: numpy array where values are inserted
        start: start index of the replacement
        values: numpy arrays with values that replace in original

    Returns:

    '''

    data = data['values'].values

    for e, v in zip(start, values):
        data[e[0]:e[0] + len(v)] = v

    return pd.Series(data)


def insert(data, start, values):
    '''

    e[0] is the starting coordinate
    e[1] is the ending coordinate

    Args:
        data: numpy array where values are inserted
        start: start index of the insertion
        values: numpy arrays to be inserted

    Returns: numpy array with values arrays inserted

    '''

    data = data['values'].values

    tmp = list(data)

    # if there are more than 1 anomalies to insert we need to check that the coordinates are corrected
    # TODO: This will fail, when the anomaly coordinates are overlapping
    if len(start) > 1:
        start_values = sorted(zip(start, values), key=lambda x: x[1][0])
        offset = 0
        for i, (coordinates, values) in enumerate(start_values):
            s, e = coordinates
            tmp = tmp[:s + offset] + list(values) + tmp[s + offset:]
            offset += e - s

    return pd.Series(tmp)
