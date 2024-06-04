"""
Test Propagation
"""
import copy

import numpy as np
import pytest
import scipy.integrate
import scipy.optimize
from conftest import loadBody

from pika.dynamics import EOMVars
from pika.dynamics.crtbp import DynamicsModel
from pika.propagate import (
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
    def test_constructor(self, emModel):
        prop = Propagator(emModel)
        assert prop.model == emModel

    def test_deepcopy(self, emModel):
        prop = Propagator(emModel)
        prop2 = copy.deepcopy(prop)

        assert id(prop.model) == id(prop2.model)
        # TODO check other attributes

    @pytest.mark.parametrize("dense", [True, False])
    @pytest.mark.parametrize(
        "eoms",
        [
            [EOMVars.STATE],
            [EOMVars.STATE, EOMVars.STM],
            [EOMVars.STATE, EOMVars.STM, EOMVars.EPOCH_DEPS, EOMVars.PARAM_DEPS],
            [EOMVars.STATE, EOMVars.STM, EOMVars.PARAM_DEPS],
            [EOMVars.STATE, EOMVars.PARAM_DEPS],
            [EOMVars.EPOCH_DEPS, EOMVars.STATE],
        ],
    )
    def test_propagate(self, emModel, dense, eoms):
        # ICs and tspan for EM L3 vertical periodic orbit
        y0 = [0.8213, 0.0, 0.5690, 0.0, -1.8214, 0.0]
        tspan = [0, 6.3111]

        prop = Propagator(emModel, dense=dense)
        sol = prop.propagate(y0, tspan, eomVars=eoms, atol=1e-12, rtol=1e-10)

        assert isinstance(sol, scipy.optimize.OptimizeResult)
        assert sol.status == 0

        assert all([tspan[0] <= t <= tspan[1] for t in sol.t])
        vecSize = emModel.stateSize(eoms)
        assert all([y.size == vecSize for y in sol.y.T])

        # Check final state value; values from old MATLAB codes
        assert pytest.approx(sol.y[0, -1], 1e-4) == 0.82129933
        assert pytest.approx(sol.y[1, -1], 1e-4) == 0.00104217
        assert pytest.approx(sol.y[2, -1], 1e-4) == 0.56899951
        assert pytest.approx(sol.y[3, -1], 1e-4) == 0.00188573
        assert pytest.approx(sol.y[4, -1], 1e-4) == -1.82139906
        assert pytest.approx(sol.y[5, -1], 1e-4) == 0.00061782

        if EOMVars.STM in eoms:
            stm = emModel.extractVars(sol.y[:, -1], EOMVars.STM)

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

        # Check for custom metadata
        assert hasattr(sol, "model")
        assert sol.model == emModel
        assert hasattr(sol, "params")
        assert sol.params == None
        assert hasattr(sol, "eomVars")
        assert np.array_equal(sorted(np.array(eoms, ndmin=1)), sol.eomVars)

    @pytest.mark.parametrize(
        "eoms", [EOMVars.STM, EOMVars.EPOCH_DEPS, EOMVars.PARAM_DEPS]
    )
    def test_propagate_invalidEOMs(self, emModel, eoms):
        y0 = [0.8213, 0.0, 0.5690, 0.0, -1.8214, 0.0]
        tspan = [0, 6.3111]

        prop = Propagator(emModel)
        with pytest.raises(RuntimeError):
            prop.propagate(y0, tspan, eomVars=eoms)

    @pytest.mark.parametrize(
        "y0, eoms",
        [
            [[0.2] * 5, EOMVars.STATE],
            [[0.3] * 43, (EOMVars.STATE, EOMVars.STM)],
        ],
    )
    def test_propagate_invalidY0(self, emModel, y0, eoms):
        tspan = [0, 6.3111]

        prop = Propagator(emModel)
        with pytest.raises(RuntimeError):
            prop.propagate(y0, tspan, eomVars=eoms)

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
        y0 = [0.8213, 0.0, 0.5690, 0.0, -1.8214, 0.0]
        tspan = [0, 6.3111]
        prop = Propagator(emModel)

        # Using VariableValueEvent to test core class behavior
        event = VariableValueEvent(*evtArgs)
        prop.events.append(event)
        sol = prop.propagate(y0, tspan)

        # Check that events were recorded
        assert isinstance(sol.t_events, list)
        assert len(sol.t_events) == 1  # 1 event in the propagation
        if event.terminal:
            assert len(sol.t_events[0]) == int(event.terminal)
        else:
            assert len(sol.t_events[0]) > 0
        assert sol.status == 0 if not event.terminal else 1


class TestAbstractEvent:
    @pytest.mark.parametrize(
        "terminal, direction",
        [[False, 0.0], [False, -3.0], [True, 2.4], [0, 1.0], [5.0, -4.0], [-1, 1]],
    )
    def test_constructor(self, terminal, direction):
        event = VariableValueEvent(0, 1.2, terminal, direction)
        assert event.terminal == terminal
        assert event.direction == direction

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
    "terminal, direction, nEvt",
    [
        # Event y = 0 should occur 5 times, 2 negative, 3 positive
        [False, 0.0, 5],
        [False, -1, 2],
        [False, 1, 3],
    ],
)
def test_variableValueEvent(emModel, terminal, direction, nEvt):
    y0 = [0.8213, 0.0, 0.5690, 0.0, -1.8214, 0.0]
    tspan = [0, 6.32]
    prop = Propagator(emModel)
    event = VariableValueEvent(1, 0.0, terminal, direction)
    prop.events.append(event)
    sol = prop.propagate(y0, tspan)

    assert len(sol.t_events) == 1  # 1 event
    t_events = sol.t_events[0]
    assert len(t_events) == nEvt
    assert sol.status == int(terminal)


@pytest.mark.parametrize("terminal, direction", [[False, 0.0]])
def test_apseEvent(emModel, terminal, direction):
    y0 = [0.8213, 0.0, 0.5690, 0.0, -1.8214, 0.0]
    tspan = [0, 6.32]
    prop = Propagator(emModel)
    prop.events.append(ApseEvent(emModel, 0, terminal, direction))
    sol = prop.propagate(y0, tspan)

    assert len(sol.t_events) == 1  # 1 event
    t_events = sol.t_events[0]
    assert len(t_events) > 0


def test_distanceEvent(emModel):
    y0 = [0.8213, 0.0, 0.5690, 0.0, -1.8214, 0.0]
    tspan = [0, 6.32]
    prop = Propagator(emModel)
    prop.events.append(DistanceEvent(0.5, [-1.0, 0.0, 0.0]))
    sol = prop.propagate(y0, tspan)

    assert len(sol.t_events) == 1  # 1 event
    t_events = sol.t_events[0]
    assert len(t_events) > 0


def test_bodyDistanceEvent(emModel):
    y0 = [0.8213, 0.0, 0.5690, 0.0, -1.8214, 0.0]
    tspan = [0, 6.32]
    prop = Propagator(emModel)
    prop.events.append(BodyDistanceEvent(emModel, 0, 1.0))
    sol = prop.propagate(y0, tspan)

    assert len(sol.t_events) == 1  # 1 event
    t_events = sol.t_events[0]
    assert len(t_events) > 0
