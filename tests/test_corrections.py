"""
Test corrections
"""
import copy
import logging

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
    ShootingProblem,
    Variable,
)
from pika.crtbp import DynamicsModel
from pika.dynamics import VarGroups
from pika.propagate import Propagator

emModel = DynamicsModel(loadBody("Earth"), loadBody("Moon"))


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

    def test_copy(self):
        var = Variable([1.0, 2.0], [True, False], "variable")
        var2 = copy.copy(var)
        assert id(var.values) == id(var2.values)
        assert id(var.name) == id(var2.name)

        # Changes to one DO affect the other
        var.values[:] = [3, 4]
        assert np.array_equal(var.values, var2.values)

        var.name = "blah"
        assert var2.name == "variable"

    def test_deepCopy(self):
        var = Variable([1.0, 2.0], [True, False], "variable")
        var2 = copy.deepcopy(var)

        assert np.array_equal(var.values, var2.values)
        assert var.name == var2.name

        # Changes to one do NOT affect the other
        var.values[:] = [3, 4]
        assert np.array_equal(var2.values, [1, 2])

        var.name = "blah"
        assert var2.name == "variable"

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
    @pytest.mark.parametrize(
        "epoch, state",
        [
            [0.0, np.arange(6)],
            [Variable(0), Variable(np.arange(6))],
        ],
    )
    @pytest.mark.parametrize("autoMask", [True, False])
    def test_constructor(self, epoch, state, autoMask):
        cp = ControlPoint(emModel, copy.deepcopy(epoch), copy.deepcopy(state), autoMask)

        assert isinstance(cp.epoch, Variable)
        # CR3BP is epoch-independent, so autoMask=True will set mask to True;
        # without autoMask, the mask will be false
        assert cp.epoch.mask == [autoMask]
        assert isinstance(cp.state, Variable)

        assert isinstance(cp.importableVars, tuple)
        assert len(cp.importableVars) == 2
        assert cp.epoch in cp.importableVars
        assert cp.state in cp.importableVars

    @pytest.mark.parametrize(
        "epoch, state",
        [
            [[0.0, 1.0], np.arange(6)],
            [0.0, np.arange(42)],
        ],
    )
    def test_constructor_errs(self, epoch, state):
        with pytest.raises(RuntimeError):
            ControlPoint(emModel, epoch, state)

    def test_fromProp(self):
        prop = Propagator(emModel)
        t0 = 0.1
        y0 = [0.8213, 0.0, 0.5690, 0.0, -1.8214, 0.0]
        sol = prop.propagate(y0, [t0, t0 + 1.2])
        cp = ControlPoint.fromProp(sol)

        assert cp.model == emModel
        assert cp.epoch.allVals[0] == t0
        assert np.array_equal(cp.state.allVals, y0)

    def test_copy(self):
        state = Variable(np.arange(6))
        epoch = Variable(0.0)
        cp = ControlPoint(emModel, epoch, state)
        cp2 = copy.copy(cp)

        assert id(cp.model) == id(cp2.model) == id(emModel)
        assert id(cp.epoch) == id(cp2.epoch) == id(epoch)
        assert id(cp.state) == id(cp2.state) == id(state)

        # Changes to variables affect both objects
        state.values[:] = np.arange(6, 12)
        assert np.array_equal(cp.state.values, cp2.state.values)

        epoch.values[:] = 3
        assert np.array_equal(cp.epoch.values, cp2.epoch.values)

    def test_deepcopy(self):
        state = Variable(np.arange(6))
        epoch = Variable(0.0)
        cp = ControlPoint(emModel, epoch, state)
        cp2 = copy.deepcopy(cp)

        assert id(cp.model) == id(cp2.model)
        assert not id(cp.epoch) == id(cp2.epoch)
        assert not id(cp.state) == id(cp2.state)

        # Changes to variables do NOT affect both objects
        state.values[:] = np.arange(6, 12)
        assert np.array_equal(cp2.state.values, np.arange(6))

        epoch.values[:] = 3
        assert np.array_equal(cp2.epoch.values, [0])


# ------------------------------------------------------------------------------
class TestSegment:
    @pytest.fixture(scope="class")
    def prop(self):
        return Propagator(emModel, dense=False)

    @pytest.fixture
    def origin(self):
        # IC for EM L3 Vertical
        return ControlPoint(emModel, 0.1, [0.8213, 0.0, 0.5690, 0.0, -1.8214, 0.0])

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
        assert id(seg.origin) == id(origin)
        assert id(seg.origin.state) == id(origin.state)
        assert id(seg.origin.epoch) == id(origin.epoch)

        assert seg.terminus == term
        if term:
            assert id(seg.terminus.state) == id(term.state)
            assert id(seg.terminus.epoch) == id(term.epoch)

        assert isinstance(seg.tof, Variable)
        assert seg.tof.values.size == 1
        assert seg.tof.allVals[0] == 1.0
        if isinstance(tof, Variable):
            assert id(seg.tof) == id(tof)

        assert isinstance(seg.prop, Propagator)
        assert not id(seg.prop) == id(_prop)  # should make a copy
        assert id(seg.prop.model) == id(origin.model)

        assert isinstance(seg.propParams, Variable)
        assert seg.propParams.values.size == 0
        if isinstance(params, Variable):
            assert id(seg.propParams) == id(params)

        assert isinstance(seg.importableVars, tuple)
        assert len(seg.importableVars) == 2
        assert seg.tof in seg.importableVars
        assert seg.propParams in seg.importableVars

    def test_constructor_altProp(self, origin, request):
        sun, earth = loadBody("Sun"), loadBody("Earth")
        model = DynamicsModel(sun, earth)
        prop = Propagator(model)
        seg = Segment(origin, 2.3, prop=prop)
        assert seg.prop.model == origin.model
        assert not seg.prop.model == model

    def test_copy(self, origin, prop):
        tof = Variable([2.1])
        terminus = ControlPoint(origin.model, 0.1, [0.0] * 6)
        params = Variable([1.0, 2.0])
        seg = Segment(origin, tof, terminus, prop, params)
        seg2 = copy.copy(seg)

        assert id(seg.origin) == id(seg2.origin) == id(origin)
        assert id(seg.terminus) == id(seg2.terminus) == id(terminus)
        assert id(seg.tof) == id(seg2.tof) == id(tof)
        assert id(seg.prop) == id(seg2.prop)
        assert id(seg.propParams) == id(seg2.propParams) == id(params)

    def test_deepcopy(self, origin, prop):
        tof = Variable([2.1])
        terminus = ControlPoint(origin.model, 0.1, [0.0] * 6)
        params = Variable([1.0, 2.0])
        seg = Segment(origin, tof, terminus, prop, params)
        seg2 = copy.deepcopy(seg)

        assert not id(seg.origin) == id(seg2.origin)
        assert id(seg.origin.model) == id(seg2.origin.model)
        assert not id(seg.terminus) == id(seg2.terminus)
        assert id(seg.terminus.model) == id(seg2.terminus.model)
        assert not id(seg.tof) == id(seg2.tof)
        assert not id(seg.prop) == id(seg2.prop)
        assert not id(seg.propParams) == id(seg2.propParams)

    @pytest.mark.parametrize(
        "varGroups",
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
    def test_propagate(self, origin, varGroups):
        seg = Segment(origin, 1.0)
        assert seg.propSol is None
        seg.propagate(varGroups)
        assert seg.propSol is not None
        assert seg.propSol.y[:, 0].size == origin.model.stateSize(varGroups)
        assert seg.propSol.t[0] == origin.epoch.allVals[0]
        assert seg.propSol.t[-1] == origin.epoch.allVals[0] + 1.0

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize(
        "varGroups1, varGroups2",
        [
            [VarGroups.STATE, VarGroups.STATE],
            [[VarGroups.STATE, VarGroups.STM], VarGroups.STATE],
        ],
    )
    def test_propagate_lazy(self, origin, mocker, varGroups1, varGroups2, lazy):
        seg = Segment(origin, 1.0)
        spy = mocker.spy(Propagator, "propagate")

        seg.propagate(varGroups1)
        assert spy.call_count == 1

        seg.propagate(varGroups2, lazy)
        assert spy.call_count == 1 if lazy else 2
        assert seg.propSol is not None
        assert seg.propSol.y[:, 0].size >= origin.model.stateSize(varGroups2)
        assert seg.propSol.t[0] == origin.epoch.allVals[0]
        assert seg.propSol.t[-1] == origin.epoch.allVals[0] + 1.0

    @pytest.mark.parametrize(
        "fcn, shapeOut",
        [
            ["state", (6,)],
            ["partials_state_wrt_time", (6,)],
            ["partials_state_wrt_initialState", (6, 6)],
            ["partials_state_wrt_epoch", (6,)],
            ["partials_state_wrt_params", (6,)],
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
        assert seg.partials_state_wrt_params().shape == (6,)
        assert seg.partials_state_wrt_epoch().shape == (6,)
        assert seg.partials_state_wrt_initialState().shape == (6, 6)
        assert seg.partials_state_wrt_time().shape == (6,)
        assert seg.state().shape == (6,)


# ------------------------------------------------------------------------------
class TestCorrectionsProblem:
    # -------------------------------------------
    # Variables

    def test_addVariables(self):
        prob = CorrectionsProblem()
        var = Variable([1.0, 2.0])
        assert prob._freeVars == []

        prob.addVariables(var)
        assert var in prob._freeVars  # successful add

    def test_addVariables_multi(self):
        prob = CorrectionsProblem()
        var1, var2 = Variable([1.0, 2.0]), Variable([3.0, 4.0])
        prob.addVariables([var1, var2])
        assert var1 in prob._freeVars
        assert var2 in prob._freeVars

    def test_addEmptyVariable(self):
        # A variable with no free values will not be added
        prob = CorrectionsProblem()
        var = Variable([1.0, 2.0], mask=[1, 1])
        prob.addVariables(var)
        assert not var in prob._freeVars
        # TODO check logging

    @pytest.mark.parametrize(
        "obj",
        [
            ControlPoint(emModel, 0, [0] * 6, autoMask=False),
            Segment(ControlPoint(emModel, 0, [0] * 6), 1.2, propParams=[1, 2, 3]),
        ],
    )
    def test_importVariables(self, obj):
        prob = CorrectionsProblem()
        prob.importVariables(obj)

        assert len(prob._freeVars) == len(obj.importableVars)
        for var in obj.importableVars:
            assert var in prob._freeVars

    @pytest.mark.parametrize(
        "variables",
        [
            [Variable([3.0, 2.0])],
            [Variable(1.0), Variable(2.0), Variable(3.0)],
        ],
    )
    def test_rmVariables(self, variables):
        prob = CorrectionsProblem()
        for var in variables:
            prob.addVariables(var)

        # remove one
        prob.rmVariables(variables[0])
        assert not variables[0] in prob._freeVars

        # check that original object is not affected
        assert all(variables[0].values.data)

    def test_rmVariables_multi(self):
        prob = CorrectionsProblem()
        var1, var2 = Variable([1.0, 2.0]), Variable([3.0, 4.0])
        prob.addVariables([var1, var2])
        prob.rmVariables([var1, var2])
        assert prob._freeVars == []

    @pytest.mark.parametrize("variables", [[], [Variable([3.0, 2.0])]])
    def test_rmVariables_missing(self, variables):
        prob = CorrectionsProblem()
        for var in variables:
            prob.addVariables(var)

        # Should do nothing if asked to remove a variable
        # TODO check logging?  caplog
        prob.rmVariables(Variable([1.2, 3.4]))

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
            prob.addVariables(var)

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
            prob.addVariables(var)

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
            prob.addVariables(var)

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
            prob.addVariables(var)

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
            prob.addVariables(var)

        assert prob.numFreeVars == sum([var.numFree for var in variables])

    # -------------------------------------------
    # Constraints

    def test_addConstraints(self):
        prob = CorrectionsProblem()
        assert prob._constraints == []  # empty to start

        var = Variable([1.1])
        con = constraints.VariableValueConstraint(var, [0.0])
        prob.addConstraints(con)
        assert con in prob._constraints  # was added

    def test_addConstraints_multi(self):
        prob = CorrectionsProblem()
        var = Variable([1.0])
        con1 = constraints.VariableValueConstraint(var, [0.0])
        con2 = constraints.VariableValueConstraint(var, [2.0])
        prob.addConstraints([con1, con2])
        assert con1 in prob._constraints
        assert con2 in prob._constraints

    def test_addEmptyConstraint(self):
        prob = CorrectionsProblem()
        var = Variable([1.0])
        con = constraints.VariableValueConstraint(var, [None])
        assert con.size == 0

        prob.addConstraints(con)
        assert not con in prob._constraints
        # TODO check logging

    def test_rmConstraints(self):
        prob = CorrectionsProblem()
        var = Variable([1.1])
        con = constraints.VariableValueConstraint(var, [0.0])
        prob.addConstraints(con)
        prob.rmConstraints(con)

        assert not con in prob._constraints

    def test_rmConstraints_multi(self):
        prob = CorrectionsProblem()
        var = Variable([1.0])
        con1 = constraints.VariableValueConstraint(var, [0.0])
        con2 = constraints.VariableValueConstraint(var, [2.0])
        prob.addConstraints([con1, con2])
        prob.rmConstraints([con1, con2])
        assert prob._constraints == []

    def test_rmConstraints_missing(self):
        prob = CorrectionsProblem()
        var = Variable([1.1])
        con = constraints.VariableValueConstraint(var, [0.0])

        # Should do nothing if asked to remove a constraint that hasn't been added
        # TODO check logging?
        prob.rmConstraints(con)

    def test_clearConstraints(self):
        prob = CorrectionsProblem()
        var = Variable([1.1])
        con = constraints.VariableValueConstraint(var, [0.0])
        prob.addConstraints(con)
        prob.clearConstraints()

        assert not con in prob._constraints

    def test_constraintIndexMap(self):
        prob = CorrectionsProblem()
        var = Variable([1.1])
        con = constraints.VariableValueConstraint(var, [0.0])
        prob.addConstraints(con)

        indexMap = prob.constraintIndexMap()
        assert isinstance(indexMap, dict)
        assert con in indexMap
        assert isinstance(indexMap[con], int)

    def test_constraintVec(self):
        prob = CorrectionsProblem()
        var = Variable([1.1])
        prob.addVariables(var)
        con = constraints.VariableValueConstraint(var, [0.0])
        prob.addConstraints(con)

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
            prob.addConstraints(con)

        assert prob.numConstraints == sum([con.size for con in cons])

    # -------------------------------------------
    # Jacobian

    def jacProb(self, posMask, posVals):
        prob = CorrectionsProblem()

        pos = Variable([1.0, 2.0, 3.0], mask=posMask)
        vel = Variable([4.0, 5.0, 6.0])

        matchPos = constraints.VariableValueConstraint(pos, posVals)
        matchDX = constraints.VariableValueConstraint(vel, [4.01, None, None])

        prob.addVariables(pos)
        prob.addVariables(vel)

        prob.addConstraints(matchPos)
        prob.addConstraints(matchDX)

        return prob

    @pytest.mark.parametrize(
        "posMask, posVals",
        [
            [[0, 0, 0], [None, 2.0, 3.0]],
            [[1, 0, 1], [None, 1.0, None]],
            [[0, 0, 1], [1.0, 2.0, None]],
        ],
    )
    def test_jacobian(self, posMask, posVals):
        prob = self.jacProb(posMask, posVals)
        jac = prob.jacobian()
        assert isinstance(jac, np.ndarray)
        assert jac.shape == (prob.numConstraints, prob.numFreeVars)

        # There should be a one in the Jacobian for each constraint
        assert sum(jac.flat) == prob.numConstraints

    @pytest.mark.parametrize(
        "posMask, posVals",
        [
            [[0, 0, 0], [None, 2.0, 3.0]],
            [[1, 0, 1], [None, 1.0, None]],
            [[0, 0, 1], [1.0, 2.0, None]],
        ],
    )
    def test_checkJacobian(self, posMask, posVals):
        prob = self.jacProb(posMask, posVals)

        # Save values for later
        freeVarVec = copy.copy(prob.freeVarVec())
        constraintVec = copy.copy(prob.constraintVec())
        jacobian = copy.copy(prob.jacobian())

        assert prob.checkJacobian(tol=1e-6)

        # Make sure original problem has not been modified
        assert np.array_equal(prob.freeVarVec(), freeVarVec)
        assert np.array_equal(prob.constraintVec(), constraintVec)
        assert np.array_equal(prob.jacobian(), jacobian)

    def test_checkJacobian_fails(self, caplog):
        prob = self.jacProb([0, 0, 0], [None, 2.0, 3.0])

        # An absurdly small tolerance will trigger failure
        with caplog.at_level(logging.DEBUG, logger="pika"):
            assert not prob.checkJacobian(tol=1e-24)

        for record in caplog.records:
            if not record.name == "pika.corrections":
                continue
            # All records should be errors
            assert record.levelno == logging.ERROR
            assert record.message.startswith("Jacobian error")

    # -------------------------------------------
    # Caching

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
        prob.addVariables(var)
        prob.addConstraints(con)
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
        self.assertCached(prob, fMap=True, cMap=True, cVec=True)

        self.clearCache(prob)
        prob.jacobian()
        self.assertCached(prob, fMap=True, cMap=True, jac=True)

        # Calling updateFreeVars clears most of the caches
        prob.updateFreeVars(np.array([2, 0, 1]))
        self.assertCached(prob, fMap=True, cMap=True)


# ------------------------------------------------------------------------------
class TestShootingProblem:
    @pytest.fixture(scope="class")
    def origin(self):
        return ControlPoint(emModel, 0.1, [0.8213, 0.0, 0.5690, 0.0, -1.8214, 0.0])

    def test_constructor(self):
        prob = ShootingProblem()
        assert prob._segments == []
        assert prob._points == []
        assert prob._adjMat is None

    def test_addSegments(self, origin):
        prob = ShootingProblem()
        seg = Segment(origin, 2.3)
        prob.addSegments(seg)
        assert seg in prob._segments

    def test_addSegments_multiple(self, origin):
        prob = ShootingProblem()
        seg = Segment(origin, 2.3)
        seg2 = Segment(origin, 1.2)
        prob.addSegments([seg, seg2])

        assert seg in prob._segments
        assert seg2 in prob._segments

    # TODO cannot add non-segment; logs error

    def test_addSegments_multiple_duplicate(self, origin):
        prob = ShootingProblem()
        seg = Segment(origin, 2.3)
        prob.addSegments([seg, seg])

        assert seg in prob._segments
        assert len(prob._segments) == 1

    def test_rmSegments(self, origin):
        prob = ShootingProblem()
        seg = Segment(origin, 2.3)
        prob.addSegments(seg)
        prob.rmSegments(seg)
        assert prob._segments == []

    def test_rmSegments_multiple(self, origin):
        prob = ShootingProblem()
        seg = Segment(origin, 2.3)
        seg2 = Segment(origin, 1.2)
        prob.addSegments([seg, seg2])
        prob.rmSegments([seg, seg2])
        assert prob._segments == []

    def test_adjacencyMatrix_fwrdTime(self):
        points = [ControlPoint(emModel, v, [v] * 6) for v in (0.0, 1.0, 2.0)]
        seg1 = Segment(points[0], 0.1, points[1])
        seg2 = Segment(points[1], 0.2, points[2])

        prob = ShootingProblem()
        prob.addSegments([seg1, seg2])
        prob.build()
        adjMat = prob.adjacencyMatrix()

        assert isinstance(adjMat, np.ndarray)
        assert np.array_equal(adjMat, [[None, 0, None], [None, None, 1], [None] * 3])

    def test_adjacencyMatrix_revTime(self):
        points = [ControlPoint(emModel, v, [v] * 6) for v in (0.0, 1.0, 2.0)]
        seg1 = Segment(points[0], -0.1, points[1])
        seg2 = Segment(points[1], -0.2, points[2])

        prob = ShootingProblem()
        prob.addSegments([seg1, seg2])
        prob.build()
        adjMat = prob.adjacencyMatrix()

        assert np.array_equal(adjMat, [[None, 0, None], [None, None, 1], [None] * 3])

    # TODO test with segments that don't have a terminus defined - probably
    #   need to catch that with a good error message at the ShooterProblem level

    @pytest.mark.parametrize("sign", [1, -1])
    def test_adjacencyMatrix_mixedTime(self, sign):
        points = [ControlPoint(emModel, v, [v] * 6) for v in (0.0, 1.0, 2.0)]
        seg1 = Segment(points[0], sign * 0.1, points[1])
        seg2 = Segment(points[0], -sign * 0.2, points[2])

        prob = ShootingProblem()
        prob.addSegments([seg1, seg2])
        prob.build()
        adjMat = prob.adjacencyMatrix()

        assert np.array_equal(adjMat, [[None, 0, 1], [None] * 3, [None] * 3])

    @pytest.mark.parametrize("sign", [1, -1])
    def test_adjacencyMatrix_invalid_doubledOrigin(self, sign):
        points = [ControlPoint(emModel, v, [v] * 6) for v in (0.0, 1.0, 2.0)]
        seg1 = Segment(points[0], sign * 0.1, points[1])
        seg2 = Segment(points[0], sign * 0.2, points[2])

        prob = ShootingProblem()
        prob.addSegments([seg1, seg2])
        with pytest.raises(RuntimeError):
            prob.build()

        adjMat = prob.adjacencyMatrix()
        assert np.array_equal(adjMat, [[None, 0, 1], [None] * 3, [None] * 3])

        errors = prob.checkValidGraph()
        assert isinstance(errors, list)
        assert len(errors) == 1
        assert "Origin control point" in errors[0]
        assert "linked to two segments" in errors[0]

    @pytest.mark.parametrize("sign", [1, -1])
    def test_adjacencyMatrix_invalid_tripleOrigin(self, sign):
        points = [ControlPoint(emModel, v, [v] * 6) for v in (0.0, 1.0, 2.0, 3.0)]
        seg1 = Segment(points[0], sign * 0.1, points[1])
        seg2 = Segment(points[0], -sign * 0.2, points[2])
        seg3 = Segment(points[0], sign * 0.3, points[3])

        prob = ShootingProblem()
        prob.addSegments([seg1, seg2, seg3])
        with pytest.raises(RuntimeError):
            prob.build()

        adjMat = prob.adjacencyMatrix()
        assert np.array_equal(
            adjMat, [[None, 0, 1, 2], [None] * 4, [None] * 4, [None] * 4]
        )

        errors = prob.checkValidGraph()
        assert len(errors) == 1
        assert "Origin control point" in errors[0]
        assert "has 3 linked segments" in errors[0]

    @pytest.mark.parametrize("sign1", [1, -1])
    @pytest.mark.parametrize("sign2", [1, -1])
    def test_adjacencyMatrix_invalid_doubledTerminus(self, sign1, sign2):
        points = [ControlPoint(emModel, v, [v] * 6) for v in (0.0, 1.0, 2.0)]
        seg1 = Segment(points[0], sign1 * 0.1, points[1])
        seg2 = Segment(points[2], sign2 * 0.2, points[1])

        prob = ShootingProblem()
        prob.addSegments([seg1, seg2])
        with pytest.raises(RuntimeError):
            prob.build()

        adjMat = prob.adjacencyMatrix()
        assert np.array_equal(adjMat, [[None, 0, None], [None] * 3, [None, 1, None]])

        errors = prob.checkValidGraph()
        assert len(errors) == 1
        assert "Terminal control point" in errors[0]

    @pytest.mark.parametrize("sign", [1, -1])
    def test_adjacencyMatrix_invalid_cycleSegment(self, sign):
        points = [ControlPoint(emModel, v, [v] * 6) for v in [0.0]]
        seg = Segment(points[0], sign * 0.1, points[0])

        prob = ShootingProblem()
        prob.addSegments(seg)
        with pytest.raises(RuntimeError):
            prob.build()

        adjMat = prob.adjacencyMatrix()
        assert np.array_equal(adjMat, [[0]])

        errors = prob.checkValidGraph()
        assert len(errors) == 1
        assert "links point 0 to itself" in errors[0]


# ------------------------------------------------------------------------------
class TestDifferentialCorrector:
    def test_simpleCorrections(self):
        # Create an initial state with velocity states free
        q0 = Variable(
            [0.8213, 0.0, 0.5690, 0.0, -1.8214, 0.0], [True] * 3 + [False] * 3
        )

        origin = ControlPoint(emModel, 0, q0)

        # Target roughly halfway around
        terminus = ControlPoint(emModel, 0, [0.82, 0.0, -0.57, 0.0, 0.0, 0.0])
        segment = Segment(origin, 3.1505, terminus)

        problem = CorrectionsProblem()
        problem.addVariables(origin.state)
        problem.addVariables(segment.tof)

        # Constrain the position at the end of the arc to match the terminus
        problem.addConstraints(
            constraints.ContinuityConstraint(segment, indices=[0, 1, 2])
        )

        corrector = DifferentialCorrector()
        solution, log = corrector.solve(problem)

        assert isinstance(solution, CorrectionsProblem)
        assert not id(solution) == id(problem)
        assert corrector.convergenceCheck.isConverged(solution)

        assert isinstance(log, dict)
        assert log["status"] == "converged"
        assert len(log["iterations"]) > 2

    def test_multipleShooter(self):
        q0 = [0.8213, 0.0, 0.5690, 0.0, -1.8214, 0.0]
        period = 6.311
        prop = Propagator(emModel, dense=False)
        sol = prop.propagate(
            q0,
            [0, period],
            t_eval=[0.0, period / 4, period / 2, 3 * period / 4, period],
        )
        points = [ControlPoint.fromProp(sol, ix) for ix in range(len(sol.t))]
        segments = [
            Segment(points[ix], period / 4, terminus=points[ix + 1], prop=prop)
            for ix in range(len(points) - 1)
        ]

        # make terminus of final arc the origin of the first
        segments[-1].terminus = points[0]

        # use single TOF for all arcs
        for seg in segments[1:]:
            seg.tof = segments[0].tof

        problem = ShootingProblem()
        problem.addSegments(segments)

        # Create continuity constraints
        for seg in segments:
            problem.addConstraints(constraints.ContinuityConstraint(seg))

        problem.build()

        # Check that variables have been added correctly
        for point in points[:-1]:
            assert point.state in problem._freeVars
            assert not point.epoch in problem._freeVars  # epoch-independent problem
        assert not points[-1].state in problem._freeVars
        assert not points[-1].epoch in problem._freeVars
        assert segments[0].tof in problem._freeVars

        # Free variables:  4x state + 1x TOF = 5
        assert len(problem._freeVars) == 5

        corrector = DifferentialCorrector()
        solution, log = corrector.solve(problem)

        assert isinstance(solution, ShootingProblem)
        assert log["status"] == "converged"
