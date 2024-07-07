"""
Test numerics module
"""
import numpy as np
import pytest

from pika import numerics


@pytest.mark.parametrize("val", [0.0, 1.0, np.pi, np.pi / 2, 5.123])
def test_derivative_sin(val):
    # Test derivative computation with simple sin function
    meta = {}
    deriv = numerics.derivative(np.sin, val, 0.25, nIter=11, meta=meta)
    assert isinstance(deriv, float)

    # Test correct computation of derivative
    trueDeriv = np.cos(val)
    assert deriv == pytest.approx(trueDeriv, rel=1e-4)

    # Test metadata output
    assert "err" in meta
    assert "count" in meta
    assert meta["count"] <= 11

    assert "tableau" in meta
    assert isinstance(meta["tableau"], np.ndarray)
    assert meta["tableau"].shape == (11, 11)


@pytest.mark.parametrize("val", [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.2, 5.4]])
def test_derivative_multivar(val):
    func = lambda x: np.asarray([np.sin(x[0]), np.cos(x[1])])
    trueDeriv = lambda x: np.asarray([[np.cos(x[0]), 0.0], [0.0, -np.sin(x[1])]])

    deriv = numerics.derivative_multivar(func, val, 0.25)
    assert isinstance(deriv, np.ndarray)
    assert deriv.shape == (2, 2)
    np.testing.assert_allclose(deriv, trueDeriv(val))
