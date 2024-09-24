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

        assert "mu" in model._properties
        assert model._properties["mu"] < 0.5
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
        assert model.stateSize(VarGroup.STATE) == 6
        assert model.stateSize(VarGroup.STM) == 36
        assert model.stateSize([VarGroup.STATE, VarGroup.STM]) == 42
        assert model.stateSize(VarGroup.EPOCH_PARTIALS) == 0
        assert model.stateSize(VarGroup.PARAM_PARTIALS) == 0

    # TODO test that modifying a property from properties fcn doesn't affect
    #   stored values

    @pytest.mark.parametrize(
        "append",
        [
            VarGroup.STATE,
            VarGroup.STM,
            VarGroup.EPOCH_PARTIALS,
            VarGroup.PARAM_PARTIALS,
        ],
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

    @pytest.mark.parametrize("ix", [-1, 2])
    def test_bodyState_invalidIx(self, ix):
        model = DynamicsModel(earth, moon)
        with pytest.raises(IndexError):
            model.bodyState(ix, 0.0)

    def test_varNames(self):
        model = DynamicsModel(earth, moon)
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

    @pytest.mark.parametrize("jit", [True, False])
    def test_checkPartials(self, jit, mocker):
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

    def test_checkPartials_fails(self):
        model = DynamicsModel(earth, moon)
        y0 = [0.8213, 0.0, 0.5690, 0.0, -1.8214, 0.0]
        y0 = model.appendICs(
            y0, [VarGroup.STM, VarGroup.EPOCH_PARTIALS, VarGroup.PARAM_PARTIALS]
        )
        tspan = [1.0, 2.0]

        # An absurdly small tolerance will trigger failure
        assert not model.checkPartials(y0, tspan, rtol=1e-24, atol=1e-24)
