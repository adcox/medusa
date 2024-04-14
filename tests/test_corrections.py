"""
Test corrections
"""

import numpy as np
import pytest
from conftest import loadBody

import pika.corrections.constraints as constraints
from pika.corrections import (
    AbstractConstraint,
    ControlPoint,
    CorrectionsProblem,
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

        assert isinstance(var.values, np.ndarray)
        assert len(var.values.shape) == 1

        assert all(var.values.data == np.array(vals, ndmin=1))
        assert all(var.values.mask == np.array(mask, ndmin=1))
        assert var.name == name

    @pytest.mark.parametrize(
        "vals, mask",
        [
            [[1.0], [False]],
            [[1.0, 2.0], [True, False]],
            [[1.0, 2.0], [True, True]],
        ],
    )
    def test_allVals(self, vals, mask):
        var = Variable(vals, mask)
        assert np.array_equal(var.allVals, vals)

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

    # TODO test ControlPoint.fromProp


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
        "tof, _prop, params",
        [
            [1.0, None, []],
            [1.0, "prop", []],
            [Variable(1.0, True), None, Variable([])],
        ],
    )
    def test_constructor(self, origin, tof, _prop, params, request):
        if not _prop is None:
            _prop = request.getfixturevalue(_prop)

        seg = Segment(origin, tof, _prop, params)

        assert seg.origin == origin

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
        seg = Segment(origin, 2.3, prop)
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


# ------------------------------------------------------------------------------
class TestCorrectionsProblem:
    # -------------------------------------------
    # Variables

    def test_addVariable(self):
        prob = CorrectionsProblem()
        var = Variable([1.0, 2.0])
        assert prob._freeVarIndexMap == {}  # empty to start
        assert prob._freeVarVec.size == 0  # empty to start

        prob.addVariable(var)
        assert var in prob._freeVarIndexMap  # successfull add
        assert prob._freeVarIndexMap[var] is None  # no index set yet
        assert prob._freeVarVec.size == 0  # not updated

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
        assert not variables[0] in prob._freeVarIndexMap

        # check that original object is not affected
        assert all(variables[0].values.data)

    @pytest.mark.parametrize("variables", [[], [Variable([3.0, 2.0])]])
    def test_rmVariable_missing(self, variables):
        prob = CorrectionsProblem()
        for var in variables:
            prob.addVariable(var)

        # Should do nothing if asked to remove a variable
        # TODO check logging?
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
        assert prob._freeVarIndexMap == {}

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

        indexMap = prob.freeVarIndexMap(False)
        for var in variables:
            assert var in indexMap
            assert indexMap[var] is None

        indexMap = prob.freeVarIndexMap(True)
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

        # Need to update variable indices before building vector
        prob.freeVarIndexMap(True)

        vec = prob.freeVarVec(False)
        assert isinstance(vec, np.ndarray)
        assert vec.size == 0

        vec = prob.freeVarVec(True)
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
    def test_numFreeVars(self, variables):
        prob = CorrectionsProblem()
        for var in variables:
            prob.addVariable(var)

        assert prob.numFreeVars == sum([var.numFree for var in variables])

    # -------------------------------------------
    # Constraints

    def test_addConstraint(self):
        prob = CorrectionsProblem()
        assert prob._constraintIndexMap == {}  # empty to start
        assert prob._constraintVec.size == 0  # empty to start

        var = Variable([1.1])
        con = constraints.VariableValueConstraint(var, [0.0])
        prob.addConstraint(con)
        assert con in prob._constraintIndexMap  # was added
        assert prob._constraintIndexMap[con] is None  # no index set yet
        assert prob._constraintVec.size == 0  # not updated

    def test_rmConstraint(self):
        prob = CorrectionsProblem()
        var = Variable([1.1])
        con = constraints.VariableValueConstraint(var, [0.0])
        prob.addConstraint(con)
        prob.rmConstraint(con)

        assert not con in prob._constraintIndexMap

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

        assert not con in prob._constraintIndexMap

    def test_constraintIndexMap(self):
        prob = CorrectionsProblem()
        var = Variable([1.1])
        con = constraints.VariableValueConstraint(var, [0.0])
        prob.addConstraint(con)

        indexMap = prob.constraintIndexMap(False)
        assert con in indexMap
        assert indexMap[con] is None

        indexMap = prob.constraintIndexMap(True)
        assert con in indexMap
        assert isinstance(indexMap[con], int)

    def test_constraintVec(self):
        prob = CorrectionsProblem()
        var = Variable([1.1])
        prob.addVariable(var)
        con = constraints.VariableValueConstraint(var, [0.0])
        prob.addConstraint(con)
        prob.freeVarIndexMap(True)  # must build indices to eval constraints
        prob.constraintIndexMap(True)
        prob.freeVarVec(True)  # must update free variables to eval constraints

        vec = prob.constraintVec(False)
        assert isinstance(vec, np.ndarray)
        assert vec.size == 0

        vec = prob.constraintVec(True)
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
