"""
Test Propagation
"""
import copy

import numpy as np
import pytest
import scipy.integrate
import scipy.optimize
from conftest import loadBody

from medusa.dynamics import VarGroup
from medusa.dynamics.crtbp import DynamicsModel, State
from medusa.propagate import (
    ApseEvent,
    BodyDistanceEvent,
    DistanceEvent,
    Propagator,
    VariableValueEvent,
)


@pytest.fixture
def emModel():
    earth = loadBody("Earth")
    moon = loadBody("Moon")
    return DynamicsModel(earth, moon)


@pytest.mark.usefixtures("emModel")
class TestPropagator:
    def test_constructor(self):
        prop = Propagator(method="abc", atol=-12.1, rtol=-1.4)
        assert prop.method == "abc"
        assert prop.atol == 12.1
        assert prop.rtol == 1.4

    def test_repr(self, emModel):
        prop = Propagator()
        assert repr(prop)  # check for failure

    @pytest.mark.parametrize("dense", [True, False])
    @pytest.mark.parametrize(
        "groups",
        [
            [VarGroup.STATE],
            [VarGroup.STATE, VarGroup.STM],
            [
                VarGroup.STATE,
                VarGroup.STM,
                VarGroup.EPOCH_PARTIALS,
                VarGroup.PARAM_PARTIALS,
            ],
            [VarGroup.STATE, VarGroup.STM, VarGroup.PARAM_PARTIALS],
            [VarGroup.STATE, VarGroup.PARAM_PARTIALS],
            [VarGroup.EPOCH_PARTIALS, VarGroup.STATE],
        ],
    )
    def test_propagate(self, emModel, dense, groups):
        # ICs and tspan for EM L3 vertical periodic orbit
        y0 = State(emModel, [0.8213, 0.0, 0.5690, 0.0, -1.8214, 0.0])
        tspan = [0, 6.3111]

        prop = Propagator(dense_output=dense)
        sol = prop.propagate(y0, tspan, varGroups=groups, atol=1e-12, rtol=1e-10)

        assert isinstance(sol, scipy.integrate._ivp.ivp.OdeResult)
        assert sol.status == 0

        assert all([tspan[0] <= t <= tspan[1] for t in sol.t])
        vecSize = y0.groupSize(groups)
        assert all([y.size == vecSize for y in sol.y.T])

        # Check for custom metadata
        assert hasattr(sol, "model")
        assert sol.model == emModel
        assert id(sol.model) == id(emModel)

        assert hasattr(sol, "states")
        assert all(isinstance(q, State) for q in sol.states)
        assert len(sol.states) == len(sol.t)
        assert all(q.center == y0.center for q in sol.states)
        assert all(q.frame == y0.frame for q in sol.states)

        assert hasattr(sol, "params")
        assert sol.params == None

        assert hasattr(sol, "varGroups")
        assert np.array_equal(sorted(np.array(groups, ndmin=1)), sol.varGroups)

        # Check final state value; values from old MATLAB codes
        assert pytest.approx(sol.states[-1][0], 1e-4) == 0.82129933
        assert pytest.approx(sol.states[-1][1], 1e-4) == 0.00104217
        assert pytest.approx(sol.states[-1][2], 1e-4) == 0.56899951
        assert pytest.approx(sol.states[-1][3], 1e-4) == 0.00188573
        assert pytest.approx(sol.states[-1][4], 1e-4) == -1.82139906
        assert pytest.approx(sol.states[-1][5], 1e-4) == 0.00061782

        if VarGroup.STM in groups:
            stm = sol.states[-1].get(VarGroup.STM)

            # Check determinant of STM is unity
            assert pytest.approx(np.linalg.det(stm)) == 1.0

            # Diagonal is ~ 1
            assert pytest.approx(stm[0, 0]) == 1.00488725
            assert pytest.approx(stm[4, 4]) == 0.96591220

            assert pytest.approx(stm[1, 0], 1e-4) == -3.3386919
            assert pytest.approx(stm[2, 0], 1e-4) == 0.0035160
            assert pytest.approx(stm[3, 0], 1e-4) == -6.0843220
            assert pytest.approx(stm[1, 4], 1e-4) == -18.242672

        if dense:
            assert isinstance(sol.sol, scipy.integrate.OdeSolution)
        else:
            assert sol.sol is None

    @pytest.mark.parametrize(
        "evtArgs",
        [
            # This y = 0 event should occur 3 or 4 times in the propagation
            [1, 0.0],  # y = 0
            [1, 0.0, True],  # terminal at first occurrence
            [1, 0.0, 3],  # terminal at three occurrences
            [1, 0.0, 1, -1],  # terminal at first negative crossing
        ],
    )
    def test_propagate_events(self, emModel, evtArgs):
        y0 = State(emModel, [0.8213, 0.0, 0.5690, 0.0, -1.8214, 0.0])
        tspan = [0, 6.3111]
        prop = Propagator()

        # Using VariableValueEvent to test core class behavior
        event = VariableValueEvent(*evtArgs)
        sol = prop.propagate(y0, tspan, events=[event])

        # Check that events were recorded
        assert event in sol.events
        assert isinstance(sol.t_events, list)
        assert len(sol.t_events) == 1  # 1 event in the propagation
        if event.terminal:
            assert len(sol.t_events[0]) == int(event.terminal)
        else:
            assert len(sol.t_events[0]) > 0
        assert sol.status == 0 if not event.terminal else 1

    def test_propagate_invalidEvents(self, emModel):
        y0 = State(emModel, [0.8213, 0.0, 0.5690, 0.0, -1.8214, 0.0])
        tspan = [0, 6.3111]
        prop = Propagator()
        with pytest.raises(TypeError):
            prop.propagate(y0, tspan, events=[emModel])

    @pytest.mark.parametrize("dense", [False, True])
    def test_denseEval(self, emModel, dense):
        y0 = State(emModel, [0.8213, 0.0, 0.5690, 0.0, -1.8214, 0.0])
        tspan = [0, 6.3111]
        prop = Propagator()

        # Do the first propagation
        sol = prop.propagate(y0, tspan, dense_output=dense)
        N = len(sol.t)

        # Now get dense output
        times = np.linspace(sol.t[0], sol.t[-1], num=500)
        sol2 = prop.denseEval(sol, times)
        np.testing.assert_array_equal(sol2.t, times)
        assert sol2.y.shape[1] == 500
        assert sol.y.shape[1] == N  # test independence
        np.testing.assert_array_equal(sol2.y[:, 0], sol.y[:, 0])
        np.testing.assert_array_equal(sol2.y[:, -1], sol.y[:, -1])


class TestAbstractEvent:
    @pytest.mark.parametrize(
        "terminal, direction",
        [[False, 0.0], [False, -3.0], [True, 2.4], [0, 1.0], [5.0, -4.0], [-1, 1]],
    )
    def test_constructor(self, terminal, direction):
        event = VariableValueEvent(0, 1.2, terminal, direction)
        assert event.terminal == terminal
        assert event.direction == direction

    def test_repr(self):
        event = VariableValueEvent(0, 1.2, True, -1)
        assert repr(event)

    @pytest.mark.parametrize(
        "terminal, direction",
        [
            [None, 0.0],
            ["True", 1],
            [False, None],
            [False, "negative"],
        ],
    )
    def test_invalidArgs(self, terminal, direction):
        with pytest.raises(TypeError):
            VariableValueEvent(0, 1.2, terminal, direction)

    @pytest.mark.parametrize(
        "terminal, direction",
        [
            [False, 0.0],
            [True, 2.4],
            [0, 1.0],
        ],
    )
    def test_assignEvalAttr(self, terminal, direction):
        event = VariableValueEvent(3, 0.0, terminal, direction)
        event.assignEvalAttr()

        assert hasattr(event.eval, "terminal")
        assert event.eval.terminal == terminal

        assert hasattr(event.eval, "direction")
        assert event.eval.direction == direction


@pytest.mark.parametrize(
    "terminal, direction, nEvt, jit",
    [
        # Event y = 0 should occur 5 times, 2 negative, 3 positive
        [False, 0.0, 5, True],
        [False, 0.0, 5, False],
        [False, -1, 2, True],
        [False, 1, 3, True],
    ],
)
def test_variableValueEvent(emModel, terminal, direction, nEvt, jit, mocker):
    if not jit:
        # Test evaluation with and without numba JIT
        mocker.patch.object(
            VariableValueEvent, "_eval", VariableValueEvent._eval.py_func
        )

    y0 = State(emModel, [0.8213, 0.0, 0.5690, 0.0, -1.8214, 0.0])
    tspan = [0, 6.32]
    prop = Propagator()
    event = VariableValueEvent(1, 0.0, terminal, direction)
    sol = prop.propagate(y0, tspan, events=[event])

    assert len(sol.t_events) == 1  # 1 event
    t_events = sol.t_events[0]
    assert len(t_events) == nEvt
    assert sol.status == int(terminal)
    for t, y in zip(sol.t_events[0], sol.y_events[0]):
        assert event.eval(t, y, [VarGroup.STATE], []) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("terminal, direction", [[False, 0.0]])
def test_apseEvent(emModel, terminal, direction):
    y0 = State(emModel, [0.8213, 0.0, 0.5690, 0.0, -1.8214, 0.0])
    tspan = [0, 6.32]
    prop = Propagator()
    event = ApseEvent(emModel, 0, terminal, direction)
    sol = prop.propagate(y0, tspan, events=[event])

    assert len(sol.t_events) == 1  # 1 event
    t_events = sol.t_events[0]
    assert len(t_events) > 0
    for t, y in zip(sol.t_events[0], sol.y_events[0]):
        assert event.eval(t, y, [VarGroup.STATE], []) == pytest.approx(0.0, abs=1e-12)


def test_distanceEvent(emModel):
    y0 = State(emModel, [0.8213, 0.0, 0.5690, 0.0, -1.8214, 0.0])
    tspan = [0, 6.32]
    prop = Propagator()
    event = DistanceEvent(0.5, [-1.0, 0.0, 0.0])
    sol = prop.propagate(y0, tspan, events=[event])

    assert len(sol.t_events) == 1  # 1 event
    t_events = sol.t_events[0]
    assert len(t_events) > 0
    for t, y in zip(sol.t_events[0], sol.y_events[0]):
        assert event.eval(t, y, [VarGroup.STATE], []) == pytest.approx(0.0, abs=1e-12)


def test_bodyDistanceEvent(emModel):
    y0 = State(emModel, [0.8213, 0.0, 0.5690, 0.0, -1.8214, 0.0])
    tspan = [0, 6.32]
    prop = Propagator()
    event = BodyDistanceEvent(emModel, 0, 1.0)
    sol = prop.propagate(y0, tspan, events=[event])

    assert len(sol.t_events) == 1  # 1 event
    t_events = sol.t_events[0]
    assert len(t_events) > 0
    for t, y in zip(sol.t_events[0], sol.y_events[0]):
        assert event.eval(t, y, [VarGroup.STATE], []) == pytest.approx(0.0, abs=1e-12)
