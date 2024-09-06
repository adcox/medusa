"""
Utilities
"""
import numpy as np


def _iterate(val):
    """
    A generator that iterates on the input, ``val``. If the input is a string,
    it is yielded without iterating.
    """
    if isinstance(val, str):
        yield val
    else:
        try:
            for item in val:
                yield item
        except TypeError:
            yield val


def toList(val):
    """
    Convert an object to a list

    Args:
        val (object): the input

    Returns:
        list: a list containing the input

    Examples:
        >>> toList(1.23)
            [1.23]
        >>> toList(np.array([1,2,3]))
            [1, 2, 3]
    """
    # Convert to list
    return list(_iterate(val))


def toArray(val):
    """
    Convert an object to a numpy array

    Args:
        val (object): the input

    Returns:
        numpy.ndarray: an array containing the input

    Examples:
        >>> toArray(1.23)
            ndarray([1.23])
    """
    # Convert to array
    return np.asarray(_iterate(val))
