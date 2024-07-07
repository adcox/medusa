"""
Test Low-Thrust control and dynamics
"""
import logging

import pytest
from conftest import loadBody

from pika import numerics
from pika.dynamics import VarGroups
from pika.lowthrust.control import *
from pika.lowthrust.dynamics import LowThrustCrtbpDynamics
from pika.propagate import Propagator

earth = loadBody("Earth")
moon = loadBody("Moon")


class TestControlTerm:
    # Basic test to check type, size, and shape of control term outputs

    @pytest.fixture(
        scope="class",
        params=[
            ["ConstThrustTerm", [2.3], {}],
            ["ConstMassTerm", [123.8], {}],
            ["ConstOrientTerm", [0.1, 2.12], {}],
        ],
        ids=["ConstThrustTerm", "ConstMassTerm", "ConstOrientTerm"],
    )
    def term(self, request):
        cls, args, kwargs = request.param[0], request.param[1], request.param[2]
        return eval(cls)(*args, **kwargs)

    @pytest.fixture
    def integArgs(self, term):
        nCore = term.coreStateSize
        t = 1.23
        y = np.arange(nCore)
        if term.numStates > 0:
            y = np.concatenate(y, np.arange(nCore, nCore + term.numStates))

        varGroups = (VarGroups.STATE,)
        term.paramIx0 = 1
        return (t, y, varGroups, [99] + term.params)

    def test_constructor(self, term):
        assert isinstance(term, ControlTerm)
        assert term.paramIx0 is None

    def test_epochIndependent(self, term):
        assert isinstance(term.epochIndependent, bool)

    def test_params(self, term):
        assert isinstance(term.params, list)

    def test_stateICs(self, term):
        assert isinstance(term.stateICs, np.ndarray)
        if term.numStates == 0:
            assert term.stateICs.size == 0
        else:
            assert term.stateICs.shape == (term.numStates,)

    def test_stateDiffEqs(self, term, integArgs):
        eqs = term.stateDiffEqs(*integArgs)
        assert isinstance(eqs, np.ndarray)
        if term.numStates == 0:
            assert eqs.size == 0
        else:
            assert eqs.shape == (term.numStates,)

    def test_evalTerm(self, term, integArgs):
        term.evalTerm(*integArgs)

    def test_partials_term_wrt_coreState(self, term, integArgs):
        partials = term.partials_term_wrt_coreState(*integArgs)
        val = term.evalTerm(*integArgs)
        sz = 1 if isinstance(val, float) else val.size

        assert isinstance(partials, np.ndarray)
        assert partials.shape == (sz, term.coreStateSize)

        # Check partials
        # TODO separate core and ctrl states
        func = lambda x: np.asarray(
            term.evalTerm(integArgs[0], x, *integArgs[2:])
        ).flatten()
        numPartials = numerics.derivative_multivar(func, integArgs[1], 1e-4)
        np.testing.assert_allclose(partials, numPartials, atol=1e-8)

    def test_partials_term_wrt_ctrlState(self, term, integArgs):
        partials = term.partials_term_wrt_ctrlState(*integArgs)
        assert isinstance(partials, np.ndarray)
        if term.numStates == 0:
            assert partials.size == 0
        else:
            val = term.evalTerm(*integArgs)
            sz = 1 if isinstance(val, float) else val.size
            assert partials.shape == (sz, term.numStates)

            # Check partials
            # TODO separate core and ctrl states
            func = lambda x: np.asarray(
                term.evalTerm(integArgs[0], x, *integArgs[2:])
            ).flatten()
            numPartials = numerics.derivative_multivar(func, integArgs[1], 1e-4)
            np.testing.assert_allclose(partials, numPartials, atol=1e-8)

    def test_partials_term_wrt_epoch(self, term, integArgs):
        partials = term.partials_term_wrt_epoch(*integArgs)
        val = term.evalTerm(*integArgs)
        sz = 1 if isinstance(val, float) else val.size

        assert isinstance(partials, np.ndarray)
        assert partials.shape == (sz, 1)
        # TODO zero if epoch idependent

        # Check partials
        func = lambda x: np.asarray(term.evalTerm(x, *integArgs[1:])).flatten()
        numPartials = numerics.derivative(func, integArgs[0], 1e-4)
        np.testing.assert_allclose(partials, numPartials, atol=1e-8)

    def test_partials_term_wrt_params(self, term, integArgs):
        partials = term.partials_term_wrt_params(*integArgs)
        val = term.evalTerm(*integArgs)
        sz = 1 if isinstance(val, float) else val.size

        assert isinstance(partials, np.ndarray)
        assert partials.shape == (sz, len(integArgs[-1]))

        # Check partials
        func = lambda x: np.asarray(term.evalTerm(*integArgs[:3], x)).flatten()
        numPartials = numerics.derivative_multivar(func, integArgs[3], 1e-4)
        np.testing.assert_allclose(partials, numPartials, atol=1e-8)

    def test_partials_coreStateDEQs_wrt_ctrlState(self, term, integArgs):
        partials = term.partials_coreStateDEQs_wrt_ctrlState(*integArgs)
        assert isinstance(partials, np.ndarray)

        if term.numStates == 0:
            assert partials.size == 0
        else:
            assert partials.shape == (term.coreStateSize, term.numStates)

    def test_partials_ctrlStateDEQs_wrt_coreState(self, term, integArgs):
        partials = term.partials_ctrlStateDEQs_wrt_coreState(*integArgs)
        assert isinstance(partials, np.ndarray)

        if term.numStates == 0:
            assert partials.size == 0
        else:
            assert partials.shape == (term.numStates, term.coreStateSize)

    def test_partials_ctrlStateDEQs_wrt_ctrlState(self, term, integArgs):
        partials = term.partials_ctrlStateDEQs_wrt_ctrlState(*integArgs)
        assert isinstance(partials, np.ndarray)

        if term.numStates == 0:
            assert partials.size == 0
        else:
            assert partials.shape == (term.numStates, term.numStates)

    def test_partials_ctrlStateDEQs_wrt_epoch(self, term, integArgs):
        partials = term.partials_ctrlStateDEQs_wrt_epoch(*integArgs)
        assert isinstance(partials, np.ndarray)

        if term.numStates == 0:
            assert partials.size == 0
        else:
            assert partials.shape == (term.numStates, 1)

    def test_partials_ctrlStateDEQs_wrt_params(self, term, integArgs):
        partials = term.partials_ctrlStateDEQs_wrt_params(*integArgs)
        assert isinstance(partials, np.ndarray)

        if term.numStates == 0 or len(integArgs[-1]) == 0:
            assert partials.size == 0
        else:
            assert partials.shape == (term.numStates, 1)


class TestForceMassOrientLaw_noStates:
    @pytest.fixture()
    def law(self):
        force = ConstThrustTerm(0.011)
        mass = ConstMassTerm(1.0)
        orient = ConstOrientTerm(np.pi / 2, 0.02)
        return ForceMassOrientLaw(force, mass, orient)

    @pytest.fixture
    def integArgs(self, law):
        nCore = law.terms[0].coreStateSize
        t = 1.23
        y = np.arange(nCore)
        if law.numStates > 0:
            y = np.concatenate(y, np.arange(nCore, nCore + law.numStates))

        varGroups = (VarGroups.STATE,)
        law.registerParams(0)
        return (t, y, varGroups, law.params)

    def test_numStates(self, law):
        assert law.numStates == 0

    def test_stateICs(self, law):
        ics = law.stateICs()
        assert isinstance(ics, np.ndarray)
        assert ics.size == 0

    def test_stateDiffEqs(self, law, integArgs):
        eqs = law.stateDiffEqs(*integArgs)
        assert isinstance(eqs, np.ndarray)
        assert eqs.size == 0

    def test_stateNames(self, law):
        # TODO
        pass

    def test_registerParams(self, law):
        assert all(term.paramIx0 is None for term in law.terms)

        law.registerParams(3)
        assert law.terms[0].paramIx0 == 3
        assert law.terms[1].paramIx0 == law.terms[0].paramIx0 + len(law.terms[0].params)
        assert law.terms[2].paramIx0 == law.terms[1].paramIx0 + len(law.terms[1].params)

    def test_params(self, law):
        params = law.params
        assert isinstance(params, np.ndarray)
        assert params.shape == (4,)

    def test_accelVec(self, law, integArgs):
        accel = law.accelVec(*integArgs)
        assert isinstance(accel, np.ndarray)
        assert accel.shape == (3, 1)

        mag = np.linalg.norm(accel)
        unit = accel / mag
        assert mag == law.terms[0].evalTerm(*integArgs) / law.terms[1].evalTerm(
            *integArgs
        )
        a, b = law.terms[2].alpha, law.terms[2].beta
        assert unit[0] == np.cos(b) * np.cos(a)
        assert unit[1] == np.cos(b) * np.sin(a)
        assert unit[2] == np.sin(b)

    def test_partials_accel_wrt_coreState(self, law, integArgs):
        partials = law.partials_accel_wrt_coreState(*integArgs)
        assert isinstance(partials, np.ndarray)
        assert partials.shape == (3, law.terms[0].coreStateSize)

        # Check partials
        # TODO separate core and ctrl states
        func = lambda x: np.asarray(
            law.accelVec(integArgs[0], x, *integArgs[2:])
        ).flatten()
        numPartials = numerics.derivative_multivar(func, integArgs[1], 1e-4)
        np.testing.assert_allclose(partials, numPartials, atol=1e-8)

    def test_partials_accel_wrt_ctrlState(self, law, integArgs):
        partials = law.partials_accel_wrt_ctrlState(*integArgs)
        assert isinstance(partials, np.ndarray)
        assert partials.size == 0

    def test_partials_accel_wrt_epoch(self, law, integArgs):
        partials = law.partials_accel_wrt_epoch(*integArgs)
        assert isinstance(partials, np.ndarray)
        assert partials.shape == (3, 1)

        # Check partials
        func = lambda x: np.asarray(law.accelVec(x, *integArgs[1:])).flatten()
        numPartials = numerics.derivative(func, integArgs[0], 1e-4)
        np.testing.assert_allclose(partials, numPartials, atol=1e-8)

    def test_partials_accel_wrt_params(self, law, integArgs):
        partials = law.partials_accel_wrt_params(*integArgs)
        assert isinstance(partials, np.ndarray)
        assert partials.shape == (3, len(law.params))

        # Check partials
        func = lambda x: np.asarray(law.accelVec(*integArgs[:3], x)).flatten()
        numPartials = numerics.derivative_multivar(func, integArgs[3], 1e-4)
        np.testing.assert_allclose(partials, numPartials, atol=1e-8)

    def test_partials_ctrlStateDEQs_wrt_coreState(self, law, integArgs):
        partials = law.partials_ctrlStateDEQs_wrt_coreState(*integArgs)
        assert isinstance(partials, np.ndarray)
        assert partials.size == 0

    def test_partials_ctrlStateDEQs_wrt_ctrlState(self, law, integArgs):
        partials = law.partials_ctrlStateDEQs_wrt_ctrlState(*integArgs)
        assert isinstance(partials, np.ndarray)
        assert partials.size == 0

    def test_partials_ctrlStateDEQs_wrt_epoch(self, law, integArgs):
        partials = law.partials_ctrlStateDEQs_wrt_epoch(*integArgs)
        assert isinstance(partials, np.ndarray)
        assert partials.size == 0

    def test_partials_ctrlStateDEQs_wrt_params(self, law, integArgs):
        partials = law.partials_ctrlStateDEQs_wrt_params(*integArgs)
        assert isinstance(partials, np.ndarray)
        assert partials.size == 0


class TestLowThrustCrtbpDynamics:
    @pytest.fixture()
    def law(self):
        force = ConstThrustTerm(0.011)
        mass = ConstMassTerm(1.0)
        orient = ConstOrientTerm(np.pi / 2, 0.02)
        return ForceMassOrientLaw(force, mass, orient)

    @pytest.fixture
    def integArgs(self, law):
        nCore = law.terms[0].coreStateSize
        t = 1.23
        y = np.arange(nCore)
        if law.numStates > 0:
            y = np.concatenate(y, np.arange(nCore, nCore + law.numStates))

        varGroups = (VarGroups.STATE,)
        law.registerParams(0)
        return (t, y, varGroups, law.params)

    @pytest.fixture
    def model(self, law):
        return LowThrustCrtbpDynamics(earth, moon, law)

    def test_epochIndependent(self, model, law):
        assert model.epochIndependent == law.epochIndependent

    @pytest.mark.parametrize(
        "grp, expect",
        [
            [VarGroups.STATE, 6],
            [VarGroups.STM, 36],
            [VarGroups.EPOCH_PARTIALS, 0],
            [VarGroups.PARAM_PARTIALS, 24],
        ],
    )
    def test_stateSize(self, model, grp, expect):
        assert model.stateSize(grp) == expect

    @pytest.mark.parametrize(
        "grp",
        [
            [VarGroups.STATE],
            [VarGroups.STATE, VarGroups.STM],
            [VarGroups.STATE, VarGroups.STM, VarGroups.EPOCH_PARTIALS],
            [
                VarGroups.STATE,
                VarGroups.STM,
                VarGroups.EPOCH_PARTIALS,
                VarGroups.PARAM_PARTIALS,
            ],
        ],
    )
    def test_propagation(self, model, grp):
        y0 = [0.8213, 0.0, 0.5690, 0.0, -1.8214, 0.0]
        if len(grp) > 1:
            y0 = model.appendICs(y0, grp[1:])

        tspan = [0, 3.15]
        prop = Propagator(model)
        sol = prop.propagate(y0, tspan, params=model.ctrlLaw.params, varGroups=grp)

    def test_checkPartials(self, model, caplog):
        y0 = [0.8213, 0.0, 0.5690, 0.0, -1.8214, 0.0]
        y0 = model.appendICs(
            y0, [VarGroups.STM, VarGroups.EPOCH_PARTIALS, VarGroups.PARAM_PARTIALS]
        )
        tspan = [1.0, 1.1]

        # TODO the integration state is getting set to NaN after the first step...
        #    Try just propagating the state and STM and see if there are issues
        with caplog.at_level(logging.DEBUG, logger="pika"):
            assert model.checkPartials(y0, tspan, params=model.ctrlLaw.params, tol=1e-4)

        for record in caplog.records:
            if not record.name == "pika.dynamics":
                continue

            assert record.levelno == logging.INFO  # no error messages
