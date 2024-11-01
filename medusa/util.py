"""
Utilities
=========

Miscellaneous utility functions.

.. autosummary::
   float_eq
   toList
   toArray

Reference
-----------

.. autofunction:: float_eq
.. autofunction:: toList
.. autofunction:: toArray
"""
from collections.abc import Sequence
from typing import Iterable

import numpy as np


def _iterate(val: object) -> Iterable:
    """
    A generator that iterates on the input, ``val``. If the input is a string,
    it is yielded without iterating.
    """
    if isinstance(val, str):
        yield val
    else:
        try:
            for item in val:  # type: ignore
                yield item
        except TypeError:
            yield val


def toList(val: object) -> list:
    """
    Convert an object to a list

    Args:
        val: the input

    Returns:
        a list containing the input

    Examples:
        >>> toList(1.23)
            [1.23]
        >>> toList(np.array([1,2,3]))
            [1, 2, 3]
    """
    # Convert to list
    return list(_iterate(val))


def toArray(val: object) -> np.ndarray:
    """
    Convert an object to a numpy array

    Args:
        val: the input

    Returns:
        an array containing the input

    Examples:
        >>> toArray(1.23)
            ndarray([1.23])
    """
    # Convert to array
    return np.asarray(list(_iterate(val)))


def float_eq(f1: float, f2: float) -> bool:
    """
    Compare two floats for equality, accepting NaN == NaN

    Args:
        f1: a float
        f2: another float

    Returns:
        whether or not the two floats are equal. If both are NaN, True is returned
    """
    if np.isnan(f1) and np.isnan(f2):
        return True
    else:
        return f1 == f2


def repr(cls: object, *attributes: str) -> str:
    """
    Simple repr method for a class with the format

      <ClassName:
        attr1: repr(attr1),
        attr2: repr(attr2),
        ...
      >

    Args:
        cls: the object
        attributes: the names of the attributes to include in the repr

    Returns:
        The repr
    """
    out = f"<{cls.__class__.__name__}:"
    for attr in attributes:
        out += "\n  {!s} = {!r},".format(attr, getattr(cls, attr))
    out += "\n>"
    return out
