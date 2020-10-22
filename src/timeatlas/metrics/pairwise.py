
def relative_error(x_0: float, x: float):
    r"""
    Compute the relative error between two values

    .. math::
       \Delta x = \frac{x_0 - x}{x}

    Args:
        x_0: float of the inferred value
        x: float of the measured value

    Returns:
        float value of the relative error

    """
    return (x_0 - x) / x


def absolute_error(x_0: float, x: float):
    r"""
    Compute the absolute error between two values

    .. math::
       \Delta x = x_0 - x

    Args:
        x_0: float of the inferred value
        x: float of the measured value

    Returns:
        float value of the absolute error

    """
    return x_0 - x
