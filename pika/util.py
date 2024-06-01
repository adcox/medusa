"""
Utilities
"""
import numpy as np


def _iterate(val):
    # Iterate on an object
    if isinstance(val, str):
        yield val
    else:
        try:
            for item in val:
                yield item
        except TypeError:
            yield val


def toList(val):
    # Convert to list
    return list(_iterate(val))


def toArray(val):
    # Convert to array
    return np.asarray(_iterate(val))
