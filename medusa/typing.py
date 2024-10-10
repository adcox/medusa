"""
Define custom types
"""
from collections.abc import Sequence
from typing import TypeVar, Union

import numpy as np
import numpy.typing as npt

#: A generic array with any data type
DT = TypeVar("DT", bound=np.generic, covariant=True)
Array = Union[Sequence[DT], npt.NDArray[DT]]

#: An array-like object of ints
IntArray = Union[Sequence[int], npt.NDArray[np.signedinteger]]

#: An array-like object of floats
FloatArray = Union[Sequence[float], Sequence[np.double], npt.NDArray[np.double]]

try:
    # Works for python 3.12+
    from typing import override  # type: ignore
except ImportError:
    from overrides import override  # type: ignore