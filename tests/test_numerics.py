"""
Test numerics module
"""
import numpy as np
import pytest

from medusa import numerics


@pytest.mark.parametrize("val", [0.0, 1.0, np.pi, np.pi / 2, 5.123])
def test_derivative_univar_scalar(val):
    # univariate function, scalar output
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


@pytest.mark.parametrize("val", [0.0, 1.0, np.pi, np.pi / 2, 5.123])
def test_derivative_univar_rowVec(val):
    # univariate function, row-vector output
    func = lambda x: np.asarray([np.sin(x), np.cos(x)])
    trueDeriv = np.asarray([np.cos(val), -np.sin(val)])

    deriv = numerics.derivative(func, val, 0.25)
    assert isinstance(deriv, np.ndarray)
    np.testing.assert_allclose(deriv, trueDeriv, rtol=1e-4, atol=1e-8)


@pytest.mark.parametrize("val", [0.0, 1.0, np.pi, np.pi / 2, 5.123])
def test_derivative_univar_colVec(val):
    # univariate function, column-vector output
    func = lambda x: np.asarray([[np.sin(x)], [np.cos(x)]])
    trueDeriv = np.asarray([[np.cos(val)], [-np.sin(val)]])

    deriv = numerics.derivative(func, val, 0.25)
    assert isinstance(deriv, np.ndarray)
    np.testing.assert_allclose(deriv, trueDeriv, rtol=1e-4, atol=1e-8)


@pytest.mark.parametrize("val", [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.2, 5.4]])
def test_derivative_multivar_scalar(val):
    # multivariate function, scalar output
    func = lambda x: np.sin(x[0]) + np.cos(x[1])
    trueDeriv = np.array([np.cos(val[0]), -np.sin(val[1])], ndmin=2)

    deriv = numerics.derivative_multivar(func, val, 0.25)
    assert isinstance(deriv, np.ndarray)
    np.testing.assert_allclose(deriv, trueDeriv, rtol=1e-4, atol=1e-8)


@pytest.mark.parametrize("val", [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.2, 5.4]])
def test_derivative_multivar_rowVec(val):
    # multivariate function, row-vecor output
    func = lambda x: np.asarray([np.sin(x[0]), np.cos(x[1])])
    trueDeriv = np.asarray([[np.cos(val[0]), 0.0], [0.0, -np.sin(val[1])]])

    deriv = numerics.derivative_multivar(func, val, 0.25)
    assert isinstance(deriv, np.ndarray)
    np.testing.assert_allclose(deriv, trueDeriv)


@pytest.mark.parametrize("val", [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.2, 5.4]])
def test_derivative_multivar_colVec(val):
    # multivariate function, column-vector output
    func = lambda x: np.asarray([[np.sin(x[0])], [np.cos(x[1])]])
    trueDeriv = np.asarray([[np.cos(val[0]), 0.0], [0.0, -np.sin(val[1])]])

    deriv = numerics.derivative_multivar(func, val, 0.25)
    assert isinstance(deriv, np.ndarray)
    np.testing.assert_allclose(deriv, trueDeriv)
