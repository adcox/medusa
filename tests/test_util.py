"""
Test the util package
"""
import numpy as np
import pytest

from medusa import util


@pytest.mark.parametrize(
    "inp, out",
    [
        (1, [1]),
        ("abc", ["abc"]),
        (1.2, [1.2]),
        ([1, 2], [1, 2]),
        (["abc", "def"], ["abc", "def"]),
        ((1,), [1]),
    ],
)
def test_toList(inp, out):
    assert util.toList(inp) == out


@pytest.mark.parametrize(
    "inp, out",
    [
        (1, [1]),
        ("abc", ["abc"]),
        (1.2, [1.2]),
        ([1, 2], [1, 2]),
        (["abc", "def"], ["abc", "def"]),
        ((1,), [1]),
    ],
)
def test_toArray(inp, out):
    outp = util.toArray(inp)
    np.testing.assert_array_equal(outp, np.asarray(out))


@pytest.mark.parametrize(
    "a, b, equal",
    [
        (1.0, 1.0, True),
        (1.0, 2.0, False),
        (1.0, np.nan, False),
        (np.nan, np.nan, True),
    ],
)
def test_floatEq(a, b, equal):
    assert util.float_eq(a, b) == equal
