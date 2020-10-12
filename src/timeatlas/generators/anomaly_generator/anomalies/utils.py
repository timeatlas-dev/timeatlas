import numpy as np

import inspect


def start_indices(maximum: int, n: int):
    '''

    e[0] is the starting coordinate
    e[1] is the ending coordinate

    Args:
        maximum: max integer to draw a number from
        n: how many numbers to draw

    Returns: starting index of the events to be inserted

    '''
    return np.random.randint(0, maximum, n)


def get_function_names(anomaly_object: object):
    """

    Args:
        anomaly_object: object containing the functions

    Returns: a list of functions excl. __init__

    """
    functions = [func[0] for func in inspect.getmembers(anomaly_object, predicate=inspect.ismethod) if
                 not func[0].startswith('__')]

    return functions
