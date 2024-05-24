"""
Test corrections
"""
import numpy as np
import pytest
from conftest import loadBody

import pika.corrections as corrections
import pika.corrections.constraints as constraints
from pika.corrections import (
    AbstractConstraint,
    ControlPoint,
    CorrectionsProblem,
    DifferentialCorrector,
    Segment,
    Variable,
)
from pika.dynamics import EOMVars
from pika.dynamics.crtbp import DynamicsModel, ModelConfig
from pika.propagate import Propagator


# ------------------------------------------------------------------------------
class TestVariable:
    @pytest.mark.parametrize(
        "vals, mask, name",
        [
            [1.0, False, "state"],
            [[1.0, 2.0], [True, False], "angle"],
        ],
    )
    def test_constructor(self, vals, mask, name):
        var = Variable(vals, mask, name)

        assert isinstance(var.values, np.ma.MaskedArray)
        assert len(var.values.shape) == 1

        assert all(var.values.data == np.array(vals, ndmin=1))
        assert all(var.values.mask == np.array(mask, ndmin=1))
        assert var.name == name

        # simple properties
        assert all(var.allVals == np.array(vals, ndmin=1))
        assert all(var.mask == np.array(mask, ndmin=1))

    @pytest.mark.parametrize(
        "vals, mask",
        [
            [[1.0], [False]],
            [[1.0, 2.0], [True, False]],
            [[1.0, 2.0], [True, True]],
        ],
    )
    def test_freeVals(self, vals, mask):
        var = Variable(vals, mask)
        assert np.array_equal(
            var.freeVals, [v for ix, v in enumerate(vals) if not mask[ix]]
        )

    @pytest.mark.parametrize("mask", [[True, True], [True, False], [False, False]])
    def test_numFree(self, mask):
        var = Variable([1.0, 2.0], mask)
        assert var.numFree == sum([not m for m in mask])

    @pytest.mark.parametrize(
        "indices, expected",
        [
            [[0, 1, 2, 3], [0, 1]],
            [[0, 1], [0]],
            [[1, 2], [0, 1]],
            [[2, 3], [1]],
            [[2, 1], [0, 1]],
            [[0, 3], []],
        ],
    )
    def test_unmaskedIndices(self, indices, expected):
        var = Variable([0.0, 1.0, 2.0, 3.0], [True, False, False, True])
        assert var.unmaskedIndices(indices) == expected


# ------------------------------------------------------------------------------
class TestControlPoint:
    @pytest.fixture(scope="class")
    def model(self):
        earth, moon = loadBody("Earth"), loadBody("Moon")
        config = ModelConfig(earth, moon)
        return DynamicsModel(config)

    @pytest.mark.parametrize(
        "epoch, state",
        [
            [0.0, np.arange(6)],
            [Variable(0), Variable(np.arange(6))],
        ],
    )
    def test_constructor(self, model, epoch, state):
        cp = ControlPoint(model, epoch, state)

        assert isinstance(cp.epoch, Variable)
        assert isinstance(cp.state, Variable)

    @pytest.mark.parametrize(
        "epoch, state",
        [
            [[0.0, 1.0], np.arange(6)],
            [0.0, np.arange(42)],
        ],
    )
    def test_constructor_errs(self, model, epoch, state):
        with pytest.raises(RuntimeError):
            ControlPoint(model, epoch, state)

    def test_fromProp(self, model):
        prop = Propagator(model)
        t0 = 0.1
        y0 = [0.8213, 0.0, 0.5690, 0.0, -1.8214, 0.0]
        sol = prop.propagate(y0, [t0, t0 + 1.2])
        cp = ControlPoint.fromProp(sol)

        assert cp.model == model
        assert cp.epoch.allVals[0] == t0
        assert np.array_equal(cp.state.allVals, y0)


# ------------------------------------------------------------------------------
class TestSegment:
    @pytest.fixture(scope="class")
    def model(self):
        earth, moon = loadBody("Earth"), loadBody("Moon")
        config = ModelConfig(earth, moon)
        return DynamicsModel(config)

    @pytest.fixture(scope="class")
    def prop(self, model):
        return Propagator(model, dense=False)

    @pytest.fixture
    def origin(self, model):
        # IC for EM L3 Vertical
        return ControlPoint(model, 0.1, [0.8213, 0.0, 0.5690, 0.0, -1.8214, 0.0])

    @pytest.mark.parametrize(
        "tof, term, _prop, params",
        [
            [1.0, None, None, []],
            [1.0, "cp", None, []],
            [1.0, None, "prop", []],
            [Variable(1.0, True), None, None, Variable([])],
        ],
    )
    def test_constructor(self, origin, tof, term, _prop, params, request):
        if not _prop is None:
            _prop = request.getfixturevalue(_prop)

        if term == "cp":
            term = ControlPoint(origin.model, 0.1, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        seg = Segment(origin, tof, term, _prop, params)

        assert seg.origin == origin
        assert seg.terminus == term

        assert isinstance(seg.tof, Variable)
        assert seg.tof.values.size == 1
        assert seg.tof.allVals[0] == 1.0

        assert isinstance(seg.prop, Propagator)
        assert seg.prop.model == origin.model

        assert isinstance(seg.propParams, Variable)
        assert seg.propParams.values.size == 0

    def test_constructor_altProp(self, origin, request):
        sun, earth = loadBody("Sun"), loadBody("Earth")
        model = DynamicsModel(ModelConfig(sun, earth))
        prop = Propagator(model)
        seg = Segment(origin, 2.3, prop=prop)
        assert seg.prop.model == origin.model
        assert not seg.prop.model == model

    @pytest.mark.parametrize(
        "eomVars",
        [
            [EOMVars.STATE],
            [EOMVars.STATE, EOMVars.STM],
            [EOMVars.STATE, EOMVars.STM, EOMVars.EPOCH_DEPS],
            [EOMVars.STATE, EOMVars.STM, EOMVars.EPOCH_DEPS, EOMVars.PARAM_DEPS],
        ],
    )
    def test_propagate(self, origin, eomVars):
        seg = Segment(origin, 1.0)
        assert seg.propSol is None
        seg.propagate(eomVars)
        assert seg.propSol is not None
        assert seg.propSol.y[:, 0].size == origin.model.stateSize(eomVars)
        assert seg.propSol.t[0] == origin.epoch.allVals[0]
        assert seg.propSol.t[-1] == origin.epoch.allVals[0] + 1.0

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize(
        "eomVars1, eomVars2",
        [
            [EOMVars.STATE, EOMVars.STATE],
            [[EOMVars.STATE, EOMVars.STM], EOMVars.STATE],
        ],
    )
    def test_propagate_lazy(self, origin, mocker, eomVars1, eomVars2, lazy):
        seg = Segment(origin, 1.0)
        spy = mocker.spy(Propagator, "propagate")

        seg.propagate(eomVars1)
        assert spy.call_count == 1

        seg.propagate(eomVars2, lazy)
        assert spy.call_count == 1 if lazy else 2
        assert seg.propSol is not None
        assert seg.propSol.y[:, 0].size >= origin.model.stateSize(eomVars2)
        assert seg.propSol.t[0] == origin.epoch.allVals[0]
        assert seg.propSol.t[-1] == origin.epoch.allVals[0] + 1.0

    @pytest.mark.parametrize(
        "fcn, shapeOut",
        [
            ["finalState", (6,)],
            ["partials_finalState_wrt_time", (6,)],
            ["partials_finalState_wrt_initialState", (6, 6)],
            ["partials_finalState_wrt_epoch", (0,)],
            ["partials_finalState_wrt_params", (0,)],
        ],
    )
    def test_getPropResults(self, origin, mocker, fcn, shapeOut):
        seg = Segment(origin, 1.0)
        spy = mocker.spy(Propagator, "propagate")
        fcn = getattr(seg, fcn)
        out = fcn()
        assert spy.call_count == 1
        assert isinstance(out, np.ndarray)
        assert out.shape == shapeOut

    def test_getMultiplePropResults(self, origin):
        seg = Segment(origin, 1.0)
        assert seg.partials_finalState_wrt_params().shape == (0,)
        assert seg.partials_finalState_wrt_epoch().shape == (0,)
        assert seg.partials_finalState_wrt_initialState().shape == (6, 6)
        assert seg.partials_finalState_wrt_time().shape == (6,)
        assert seg.finalState().shape == (6,)


# ------------------------------------------------------------------------------
class TestCorrectionsProblem:
    # -------------------------------------------
    # Variables

    def test_addVariable(self):
        prob = CorrectionsProblem()
        var = Variable([1.0, 2.0])
        assert prob._freeVars == []

        prob.addVariable(var)
        assert var in prob._freeVars  # successful add

    def test_addEmptyVariable(self):
        # A variable with no free values will not be added
        prob = CorrectionsProblem()
        var = Variable([1.0, 2.0], mask=[1, 1])
        prob.addVariable(var)
        assert not var in prob._freeVars
        # TODO check logging

    @pytest.mark.parametrize(
        "variables",
        [
            [Variable([3.0, 2.0])],
            [Variable(1.0), Variable(2.0), Variable(3.0)],
        ],
    )
    def test_rmVariable(self, variables):
        prob = CorrectionsProblem()
        for var in variables:
            prob.addVariable(var)

        # remove one
        prob.rmVariable(variables[0])
        assert not variables[0] in prob._freeVars

        # check that original object is not affected
        assert all(variables[0].values.data)

    @pytest.mark.parametrize("variables", [[], [Variable([3.0, 2.0])]])
    def test_rmVariable_missing(self, variables):
        prob = CorrectionsProblem()
        for var in variables:
            prob.addVariable(var)

        # Should do nothing if asked to remove a variable
        # TODO check logging?  caplog
        prob.rmVariable(Variable([1.2, 3.4]))

    @pytest.mark.parametrize(
        "variables",
        [
            [Variable([3.0, 2.0])],
            [Variable(1.0), Variable(2.0), Variable(3.0)],
        ],
    )
    def test_clearVariables(self, variables):
        prob = CorrectionsProblem()
        for var in variables:
            prob.addVariable(var)

        prob.clearVariables()
        assert prob._freeVars == []

        # Check that original objects are not affected
        for var in variables:
            assert all(var.values.data)

    @pytest.mark.parametrize(
        "variables",
        [
            [Variable([3.0, 2.0])],
            [Variable(1.0), Variable(2.0), Variable(3.0)],
        ],
    )
    def test_freeVarIndexMap(self, variables):
        prob = CorrectionsProblem()
        for var in variables:
            prob.addVariable(var)

        indexMap = prob.freeVarIndexMap()
        assert isinstance(indexMap, dict)
        for var in variables:
            assert var in indexMap
            assert isinstance(indexMap[var], int)

    @pytest.mark.parametrize(
        "variables",
        [
            [Variable([1.0, 2.0])],
            [Variable([1.0, 2.0], [True, False])],
            [Variable([1.0, 3.0]), Variable([2.0, 4.0])],
            [Variable([1.0, 3.0], [True, False]), Variable([2.0, 4.0])],
        ],
    )
    def test_variableVec(self, variables):
        prob = CorrectionsProblem()
        for var in variables:
            prob.addVariable(var)

        vec = prob.freeVarVec()
        assert isinstance(vec, np.ndarray)
        assert vec.size == sum([var.numFree for var in variables])

    @pytest.mark.parametrize(
        "variables",
        [
            [Variable([1.0, 2.0])],
            [Variable([1.0, 2.0], [True, False])],
            [Variable([1.0, 3.0]), Variable([2.0, 4.0])],
            [Variable([1.0, 3.0], [True, False]), Variable([2.0, 4.0])],
        ],
    )
    def test_updateFreeVars(self, variables):
        prob = CorrectionsProblem()
        for var in variables:
            prob.addVariable(var)

        vec0 = prob.freeVarVec()
        newVec = np.random.rand(*vec0.shape)
        prob.updateFreeVars(newVec)

        assert prob._freeVarVec is None
        assert prob._constraintVec is None
        assert prob._jacobian is None
        assert np.array_equal(prob.freeVarVec(), newVec)

        # Check that variable objects were updated
        for var, ix0 in prob.freeVarIndexMap().items():
            assert np.array_equal(
                var.values[~var.mask], prob._freeVarVec[ix0 : ix0 + var.numFree]
            )
            # TODO check that masked elements were NOT changed

    @pytest.mark.parametrize(
        "variables",
        [
            [Variable([1.0, 2.0])],
            [Variable([1.0, 2.0], [True, False])],
            [Variable([1.0, 3.0]), Variable([2.0, 4.0])],
            [Variable([1.0, 3.0], [True, False]), Variable([2.0, 4.0])],
        ],
    )
    def test_numFreeVars(self, variables):
        prob = CorrectionsProblem()
        for var in variables:
            prob.addVariable(var)

        assert prob.numFreeVars == sum([var.numFree for var in variables])

    # -------------------------------------------
    # Constraints

    def test_addConstraint(self):
        prob = CorrectionsProblem()
        assert prob._constraints == []  # empty to start

        var = Variable([1.1])
        con = constraints.VariableValueConstraint(var, [0.0])
        prob.addConstraint(con)
        assert con in prob._constraints  # was added

    def test_addEmptyConstraint(self):
        prob = CorrectionsProblem()
        var = Variable([1.0])
        con = constraints.VariableValueConstraint(var, [None])
        assert con.size == 0

        prob.addConstraint(con)
        assert not con in prob._constraints
        # TODO check logging

    def test_rmConstraint(self):
        prob = CorrectionsProblem()
        var = Variable([1.1])
        con = constraints.VariableValueConstraint(var, [0.0])
        prob.addConstraint(con)
        prob.rmConstraint(con)

        assert not con in prob._constraints

    def test_rmConstraint_missing(self):
        prob = CorrectionsProblem()
        var = Variable([1.1])
        con = constraints.VariableValueConstraint(var, [0.0])

        # Should do nothing if asked to remove a constraint that hasn't been added
        # TODO check logging?
        prob.rmConstraint(con)

    def test_clearConstraints(self):
        prob = CorrectionsProblem()
        var = Variable([1.1])
        con = constraints.VariableValueConstraint(var, [0.0])
        prob.addConstraint(con)
        prob.clearConstraints()

        assert not con in prob._constraints

    def test_constraintIndexMap(self):
        prob = CorrectionsProblem()
        var = Variable([1.1])
        con = constraints.VariableValueConstraint(var, [0.0])
        prob.addConstraint(con)

        indexMap = prob.constraintIndexMap()
        assert isinstance(indexMap, dict)
        assert con in indexMap
        assert isinstance(indexMap[con], int)

    def test_constraintVec(self):
        prob = CorrectionsProblem()
        var = Variable([1.1])
        prob.addVariable(var)
        con = constraints.VariableValueConstraint(var, [0.0])
        prob.addConstraint(con)

        vec = prob.constraintVec()
        assert isinstance(vec, np.ndarray)
        assert vec.size == con.size
        assert all([isinstance(val, float) for val in vec])

    @pytest.mark.parametrize(
        "cons",
        [
            constraints.VariableValueConstraint(Variable([1.0]), [1.0]),
            [
                constraints.VariableValueConstraint(Variable([1.0]), [1.0]),
                constraints.VariableValueConstraint(Variable([2.0]), [2.1]),
            ],
        ],
    )
    def test_numConstraints(self, cons):
        cons = np.array(cons, ndmin=1)
        prob = CorrectionsProblem()
        for con in cons:
            prob.addConstraint(con)

        assert prob.numConstraints == sum([con.size for con in cons])

    # -------------------------------------------
    # Jacobian

    @pytest.mark.parametrize(
        "posMask, posVals",
        [
            [[0, 0, 0], [None, 2.0, 3.0]],
            [[1, 0, 1], [1.0]],
            [[0, 0, 1], [1.0, 2.0]],
        ],
    )
    def test_jacobian(self, posMask, posVals):
        prob = CorrectionsProblem()

        pos = Variable([1.0, 2.0, 3.0], mask=posMask)
        vel = Variable([4.0, 5.0, 6.0])

        matchY = constraints.VariableValueConstraint(pos, posVals)
        matchDX = constraints.VariableValueConstraint(vel, [4.01, None, None])

        prob.addVariable(pos)
        prob.addVariable(vel)

        prob.addConstraint(matchY)
        prob.addConstraint(matchDX)

        jac = prob.jacobian()
        assert isinstance(jac, np.ndarray)
        assert jac.shape == (prob.numConstraints, prob.numFreeVars)

        # There should be a one in the Jacobian for each constraint
        assert sum(jac.flat) == prob.numConstraints

    def assertCached(
        self, prob, fVec=False, fMap=False, cVec=False, cMap=False, jac=False
    ):
        flags = {
            "_freeVarVec": fVec,
            "_freeVarIndexMap": fMap,
            "_constraintVec": cVec,
            "_constraintIndexMap": cMap,
            "_jacobian": jac,
        }
        for attr, flag in flags.items():
            if flag:
                assert getattr(prob, attr) is not None, f"{attr} should not be None"
            else:
                assert getattr(prob, attr) is None, f"{attr} should be None"

    def clearCache(self, prob):
        prob._freeVarVec = None
        prob._freeVarIndexMap = None
        prob._constraintVec = None
        prob._constraintIndexMap = None
        prob._jacobian = None

    def test_caching(self):
        prob = CorrectionsProblem()
        var = Variable([0, 1, 2])
        con = constraints.VariableValueConstraint(var, [None, 1.1, None])
        prob.addVariable(var)
        prob.addConstraint(con)
        self.assertCached(prob)  # all caches are empty

        prob.freeVarIndexMap()
        self.assertCached(prob, fMap=True)

        self.clearCache(prob)
        prob.freeVarVec()
        self.assertCached(prob, fMap=True, fVec=True)

        self.clearCache(prob)
        prob.constraintIndexMap()
        self.assertCached(prob, cMap=True)

        self.clearCache(prob)
        prob.constraintVec()
        self.assertCached(prob, fMap=True, fVec=True, cMap=True, cVec=True)

        self.clearCache(prob)
        prob.jacobian()
        self.assertCached(prob, fMap=True, fVec=True, cMap=True, jac=True)

        # Calling updateFreeVars clears most of the caches
        prob.updateFreeVars(np.array([2, 0, 1]))
        self.assertCached(prob, fMap=True, cMap=True)


class TestDifferentialCorrector:
    @pytest.fixture(scope="class")
    def model(self):
        earth, moon = loadBody("Earth"), loadBody("Moon")
        return DynamicsModel(ModelConfig(earth, moon))

    def test_singleShooter(self, model):
        # Create an initial state with velocity states free
        q0 = Variable(
            [0.8213, 0.0, 0.5690, 0.0, -1.8214, 0.0], [True] * 3 + [False] * 3
        )
        origin = ControlPoint(model, 0, q0)

        # Target roughly halfway around
        terminus = ControlPoint(model, 0, [0.82, 0.0, -0.57, 0.0, 0.0, 0.0])
        segment = Segment(origin, 3.1505, terminus)

        problem = CorrectionsProblem()
        problem.addVariable(origin.state)
        problem.addVariable(segment.tof)

        # Constrain the position at the end of the arc to match the terminus
        problem.addConstraint(
            constraints.ContinuityConstraint(segment, indices=[0, 1, 2])
        )

        corrector = DifferentialCorrector()
        corrector.updateGenerator = corrections.minimumNormUpdate
        corrector.convergenceCheck = corrections.constraintVecL2Norm
        solution = corrector.solve(problem)

        assert isinstance(solution, CorrectionsProblem)
        # TODO check that constraints are met, etc.
