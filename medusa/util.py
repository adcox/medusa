"""
Utilities
=========

Miscellaneous utility functions.

.. autosummary::
   float_eq
   toList
   toArray

.. autofunction:: float_eq
.. autofunction:: toList
.. autofunction:: toArray
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
    return np.asarray(list(_iterate(val)))


def float_eq(f1, f2):
    """
    Compare two floats for equality, accepting NaN == NaN

    Args:
        f1 (float)
        f2 (float)

    Returns:
        bool: whether or not the two floats are equal. If both are NaN, True is
        returned
    """
    if np.isnan(f1) and np.isnan(f2):
        return True
    else:
        return f1 == f2
