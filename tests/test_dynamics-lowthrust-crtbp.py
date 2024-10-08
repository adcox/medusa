"""
Test the dynamics.lowthrust.crtbp  package
"""
import numpy as np
import pytest
from conftest import loadBody

from medusa.dynamics import VarGroup
from medusa.dynamics.lowthrust import (
    ConstMassTerm,
    ConstOrientTerm,
    ConstThrustTerm,
    ForceMassOrientLaw,
)
from medusa.dynamics.lowthrust.crtbp import DynamicsModel
from medusa.propagate import Propagator

earth = loadBody("Earth")
moon = loadBody("Moon")


class TestDynamicsModel:
    @pytest.fixture()
    def law(self):
        force = ConstThrustTerm(0.011)
        mass = ConstMassTerm(1.0)
        orient = ConstOrientTerm(np.pi / 2, 0.02)
        return ForceMassOrientLaw(force, mass, orient)

    @pytest.fixture
    def integArgs(self, law):
        nCore = law.terms[0]._coreStateSize
        t = 1.23
        y = np.arange(nCore)
        if law.numStates > 0:
            y = np.concatenate(y, np.arange(nCore, nCore + law.numStates))

        varGroups = (VarGroup.STATE,)
        return (t, y, varGroups, law.params)

    @pytest.fixture
    def model(self, law):
        return DynamicsModel(earth, moon, law)

    def test_epochIndependent(self, model, law):
        assert model.epochIndependent == law.epochIndependent

    @pytest.mark.parametrize(
        "grp, expect",
        [
            [VarGroup.STATE, 6],
            [VarGroup.STM, 36],
            [VarGroup.EPOCH_PARTIALS, 0],
            [VarGroup.PARAM_PARTIALS, 24],
        ],
    )
    def test_groupSize(self, model, grp, expect):
        assert model.groupSize(grp) == expect

    @pytest.mark.parametrize(
        "grp",
        [
            [VarGroup.STATE],
            [VarGroup.STATE, VarGroup.STM],
            [VarGroup.STATE, VarGroup.STM, VarGroup.EPOCH_PARTIALS],
            [
                VarGroup.STATE,
                VarGroup.STM,
                VarGroup.EPOCH_PARTIALS,
                VarGroup.PARAM_PARTIALS,
            ],
        ],
    )
    def test_propagation(self, model, grp):
        y0 = [0.8213, 0.0, 0.5690, 0.0, -1.8214, 0.0]
        if len(grp) > 1:
            y0 = model.appendICs(y0, grp[1:])

        tspan = [0, 3.15]
        prop = Propagator(model)
        prop.propagate(y0, tspan, params=model.ctrlLaw.params, varGroups=grp)

    def test_checkPartials(self, model):
        y0 = [0.8213, 0.0, 0.5690, 0.0, -1.8214, 0.0]
        y0 = model.appendICs(
            y0, [VarGroup.STM, VarGroup.EPOCH_PARTIALS, VarGroup.PARAM_PARTIALS]
        )
        tspan = [1.0, 1.1]

        assert model.checkPartials(y0, tspan, params=model.ctrlLaw.params, rtol=1e-4)

    def test_equal(self, model, law):
        from medusa.dynamics.crtbp import DynamicsModel as CRTBP

        crtbp = CRTBP(earth, moon)
        model2 = DynamicsModel(earth, moon, law)

        assert model == model2
        assert not model == crtbp
