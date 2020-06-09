import numpy as np


def start_indices(maximum, n):
    '''

    e[0] is the starting coordinate
    e[1] is the ending coordinate

    Args:
        maximum: max integer to draw a number from
        n: how many numbers to draw

    Returns: starting index of the events to be inserted

    '''
    return np.random.randint(0, maximum, n)
