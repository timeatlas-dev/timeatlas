
def relative_error(x: float, y: float):
    """
    Compute the relative error between two values in a numerically stable way.

    TODO: Should be improved with a definition such as the one in
        https://mathworld.wolfram.com/RelativeError.html

    Args:
        x: float of the true value
        y: float of the estimated value

    Returns:
        float value of the relative error

    """
    return abs(x-y)/(abs(x)+1)
