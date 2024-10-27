"""
Test CRTBP dynamics
"""
import logging

import numpy as np
import pytest
from conftest import loadBody

from medusa.data import Body
from medusa.dynamics import VarGroup
from medusa.dynamics.crtbp import DynamicsModel
from medusa.units import kg, km, sec

earth = loadBody("Earth")
moon = loadBody("Moon")
sun = loadBody("Sun")


@pytest.fixture
def model_mu():
    # A fixed mu value for unit tests that rely on output values
    model = DynamicsModel(earth, moon)
    model.mu = 0.012150584269940356
    return model


@pytest.fixture
def model():
    return DynamicsModel(earth, moon)


@pytest.mark.parametrize("bodies", [[earth, moon], [moon, earth]])
def test_bodyOrder(bodies):
    model = DynamicsModel(*bodies)

    assert isinstance(model.bodies, tuple)
    assert len(model.bodies) == 2
    assert model.bodies[0].gm > model.bodies[1].gm

    assert model.massRatio < 0.5
    assert model.charL > 1e5 * km
    assert model.charM > 1e5 * kg
    assert model.charT > 1e5 * sec


def test_equals():
    model1 = DynamicsModel(earth, moon)
    model2 = DynamicsModel(moon, earth)
    model3 = DynamicsModel(earth, sun)
    model4 = DynamicsModel(earth, moon)
    model4.massRatio *= 1.0000001

    assert model1 == model1
    assert model1 == model2
    assert not model1 == model3
    assert not model1 == model4


def test_groupSize():
    model = DynamicsModel(earth, moon)
    assert model.groupSize(VarGroup.STATE) == 6
    assert model.groupSize(VarGroup.STM) == 36
    assert model.groupSize([VarGroup.STATE, VarGroup.STM]) == 42
    assert model.groupSize(VarGroup.EPOCH_PARTIALS) == 0
    assert model.groupSize(VarGroup.PARAM_PARTIALS) == 0


@pytest.mark.parametrize(
    "append",
    [
        VarGroup.STATE,
        VarGroup.STM,
        VarGroup.EPOCH_PARTIALS,
        VarGroup.PARAM_PARTIALS,
    ],
)
def test_appendICs(model, append):
    q0 = np.array([0, 1, 2, 3, 4, 5])
    q0_mod = model.appendICs(q0, append)

    assert q0_mod.shape == (q0.size + model.groupSize(append),)


@pytest.mark.parametrize("ix", [0, 1])
def test_bodyState(model, ix):
    state = model.bodyState(ix, 0.0)
    assert isinstance(state, np.ndarray)
    assert state.shape == (6,)


@pytest.mark.parametrize("ix", [-1, 2])
def test_bodyState_invalidIx(model, ix):
    with pytest.raises(IndexError):
        model.bodyState(ix, 0.0)


def test_varNames(model):
    stateNames = model.varNames(VarGroup.STATE)
    assert stateNames == ["x", "y", "z", "dx", "dy", "dz"]

    stmNames = model.varNames(VarGroup.STM)
    assert stmNames[1] == "STM(0,1)"
    assert stmNames[35] == "STM(5,5)"
    assert stmNames[6] == "STM(1,0)"

    epochNames = model.varNames(VarGroup.EPOCH_PARTIALS)
    assert epochNames == []

    paramNames = model.varNames(VarGroup.PARAM_PARTIALS)
    assert paramNames == []


@pytest.mark.parametrize("N, transpose", [(1, False), (2, False), (2, True)])
def test_toBaseUnits(model, N, transpose):
    q0_nd = [0.64260, 0.0, 0.75004, 0.0, 0.35068, 0.0]
    q0_dim = [q * model.charL for q in q0_nd[:3]] + [
        q * model.charL / model.charT for q in q0_nd[3:]
    ]

    qIn = np.asarray([q0_nd for ix in range(N)])
    qOut = np.array([q0_dim for ix in range(N)], dtype=object)
    if transpose:
        qIn = qIn.T

    q_dim = model.toBaseUnits(qIn, VarGroup.STATE)
    assert q_dim.shape == qOut.shape
    for out, expect in zip(q_dim.flat, qOut.flat):
        assert abs((out - expect).to_base_units().magnitude) < 1e-12


@pytest.mark.parametrize("N, transpose", [(1, False), (2, False), (2, True)])
def test_normalize(model, N, transpose):
    q0_nd = [0.64260, 0.0, 0.75004, 0.0, 0.35068, 0.0]
    q0_dim = [q * model.charL for q in q0_nd[:3]] + [
        q * model.charL / model.charT for q in q0_nd[3:]
    ]

    qIn = np.asarray([q0_dim for ix in range(N)], dtype=object)
    qOut = np.array([q0_nd for ix in range(N)])
    if transpose:
        qIn = qIn.T

    q_nd = model.normalize(qIn, VarGroup.STATE)
    assert q_nd.shape == qOut.shape
    for out, expect in zip(q_nd.flat, qOut.flat):
        assert abs(out - expect) < 1e-12


@pytest.mark.parametrize("jit", [True, False])
def test_checkPartials(jit, mocker):
    # Test EOM evaluation both with and without numba JIT
    if not jit:
        mocker.patch.object(DynamicsModel, "_eoms", DynamicsModel._eoms.py_func)

    model = DynamicsModel(earth, moon)
    y0 = [0.8213, 0.0, 0.5690, 0.0, -1.8214, 0.0]
    y0 = model.appendICs(
        y0, [VarGroup.STM, VarGroup.EPOCH_PARTIALS, VarGroup.PARAM_PARTIALS]
    )
    tspan = [1.0, 2.0]
    assert model.checkPartials(y0, tspan)


def test_checkPartials_fails(model):
    y0 = [0.8213, 0.0, 0.5690, 0.0, -1.8214, 0.0]
    y0 = model.appendICs(
        y0, [VarGroup.STM, VarGroup.EPOCH_PARTIALS, VarGroup.PARAM_PARTIALS]
    )
    tspan = [1.0, 2.0]

    # An absurdly small tolerance will trigger failure
    assert not model.checkPartials(y0, tspan, rtol=1e-24, atol=1e-24)


def test_jacobi(model_mu):
    assert model_mu.jacobi([0.1, 0.2, 0.3, 0.4, -0.2, 0.01]) == pytest.approx(
        5.10758625756296, abs=1e-9
    )


def test_pseudopotential(model_mu):
    assert model_mu.pseudopotential([0.1, 0.2, 0.3]) == pytest.approx(
        2.65384312878148, abs=1e-9
    )


def test_pseudopotentialPartials(model_mu):
    grad = model_mu.partials_pseudopot_wrt_position([0.1, 0.2, 0.3])
    assert isinstance(grad, np.ndarray)
    assert grad.shape == (1, 3)

    grad_expect = np.asarray(
        [[-1.94559476692505, -3.47256751240501, -5.50885126860751]]
    )
    np.testing.assert_array_almost_equal(grad, grad_expect, decimal=9)


@pytest.mark.parametrize("bodies", [[earth, moon], [sun, earth]])
def test_equilbria(bodies):
    model = DynamicsModel(*bodies)
    eqPts = model.equilibria(tol=1e-12)

    assert isinstance(eqPts, np.ndarray)
    assert eqPts.shape == (5, 3)

    # Pseudopotential gradient is zero at equilibria
    zero = np.zeros((1, 3))
    for pt in eqPts:
        grad = model.partials_pseudopot_wrt_position(pt)
        np.testing.assert_array_almost_equal(grad, zero, decimal=11)

    # Order is L1, L2, L3, L4, L5
    mu = model.massRatio
    assert 0 < eqPts[0, 0] < 1 - mu
    assert 1 - mu < eqPts[1, 0] <= 2
    assert eqPts[2, 0] < 0
    for ix in range(3):
        assert eqPts[ix, 1] == 0.0  # should be exact
    assert eqPts[3, 0] == eqPts[4, 0]
    assert eqPts[3, 1] > 0
    assert eqPts[4, 1] < 0
