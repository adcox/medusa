"""
Test CRTBP dynamics
"""
import numpy as np
import pytest
from conftest import loadBody

from pika.crtbp import DynamicsModel
from pika.data import Body
from pika.dynamics import EOMVars

earth = loadBody("Earth")
moon = loadBody("Moon")
sun = loadBody("Sun")


class TestDynamicsModel:
    @pytest.mark.parametrize("bodies", [[earth, moon], [moon, earth]])
    def test_bodyOrder(self, bodies):
        model = DynamicsModel(*bodies)

        assert isinstance(model.bodies, tuple)
        assert len(model.bodies) == 2
        assert model.bodies[0].gm > model.bodies[1].gm

        assert "mu" in model.params
        assert model.params["mu"] < 0.5
        assert model.charL > 1.0
        assert model.charM > 1.0
        assert model.charT > 1.0

    def test_equals(self):
        model1 = DynamicsModel(earth, moon)
        model2 = DynamicsModel(moon, earth)
        model3 = DynamicsModel(earth, sun)

        assert model1 == model1
        assert model1 == model2
        assert not model1 == model3

    def test_stateSize(self):
        model = DynamicsModel(earth, moon)
        assert model.stateSize(EOMVars.STATE) == 6
        assert model.stateSize(EOMVars.STM) == 36
        assert model.stateSize([EOMVars.STATE, EOMVars.STM]) == 42
        assert model.stateSize(EOMVars.EPOCH_DEPS) == 0
        assert model.stateSize(EOMVars.PARAM_DEPS) == 0

    @pytest.mark.parametrize(
        "append", [EOMVars.STATE, EOMVars.STM, EOMVars.EPOCH_DEPS, EOMVars.PARAM_DEPS]
    )
    def test_appendICs(self, append):
        model = DynamicsModel(earth, moon)
        q0 = np.array([0, 1, 2, 3, 4, 5])
        q0_mod = model.appendICs(q0, append)

        assert q0_mod.shape == (q0.size + model.stateSize(append),)

    @pytest.mark.parametrize("ix", [0, 1])
    def test_bodyState(self, ix):
        model = DynamicsModel(earth, moon)
        state = model.bodyState(ix, 0.0)
        assert isinstance(state, np.ndarray)
        assert state.shape == (6,)
