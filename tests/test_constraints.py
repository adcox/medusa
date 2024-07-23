"""
Test Constraint objects
"""
import numpy as np
import pytest
from conftest import loadBody

import pika.corrections.constraints as pcons
from pika.corrections import (
    ControlPoint,
    CorrectionsProblem,
    Segment,
    ShootingProblem,
    Variable,
)
from pika.crtbp import DynamicsModel as crtbpModel

emModel = crtbpModel(loadBody("Earth"), loadBody("Moon"))


class TestContinuityConstraint:
    @pytest.fixture
    def origin(self, originMask):
        # IC for EM L3 Vertical
        state = Variable([0.8213, 0.0, 0.5690, 0.0, -1.8214, 0.0], originMask)
        return ControlPoint(emModel, 0.1, state)

    @pytest.fixture
    def terminus(self, terminusMask):
        state = Variable([0.0] * 6, terminusMask)
        return ControlPoint(emModel, 1.1, state)

    @pytest.fixture
    def segment(self, origin, terminus):
        return Segment(origin, 1.0, terminus=terminus)

    @pytest.fixture
    def problem(self, segment):
        prob = CorrectionsProblem()
        prob.addVariables(segment.origin.state)
        prob.addVariables(segment.terminus.state)
        prob.addVariables(segment.tof)
        prob.addVariables(segment.propParams)

        return prob

    @pytest.mark.parametrize(
        "originMask, terminusMask, indices",
        [
            [None, None, None],
            [None, None, [0, 1, 2]],
            [None, [0, 0, 0, 1, 1, 1], [0, 1, 2]],
        ],
    )
    def test_constructor(self, segment, indices):
        con = pcons.ContinuityConstraint(segment, indices=indices)

    @pytest.mark.parametrize(
        "originMask, terminusMask, indices",
        [
            [None, [0, 1, 0, 0, 0, 0], [0, 1, 1]],
        ],
    )
    def test_constructor_err(self, segment, indices):
        with pytest.raises(RuntimeError):
            pcons.ContinuityConstraint(segment, indices=indices)

    @pytest.mark.parametrize(
        "originMask, terminusMask, indices",
        [
            [None, None, None],
            [None, None, [0, 1, 2]],
            [None, [0, 0, 0, 1, 1, 1], [0, 1, 2]],
            [[0, 0, 0, 1, 1, 1], None, None],
            [[0, 1, 0, 1, 1, 1], None, None],
            [[0, 0, 0, 1, 1, 1], None, [0, 1, 2]],
            [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1], [0, 1, 2]],
            [None, [0, 1, 0, 0, 0, 0], None],
            [None, [0, 1, 0, 0, 0, 0], [0, 1, 2]],
        ],
    )
    def test_evaluate(self, segment, indices, problem):
        con = pcons.ContinuityConstraint(segment, indices=indices)
        err = con.evaluate(problem.freeVarIndexMap())
        assert isinstance(err, np.ndarray)
        assert err.size == con.size

    @pytest.mark.parametrize(
        "originMask, terminusMask, indices",
        [
            [None, None, None],
            [None, None, [0, 1, 2]],
            [None, [0, 0, 0, 1, 1, 1], [0, 1, 2]],
            [[0, 0, 0, 1, 1, 1], None, None],
            [[0, 1, 0, 1, 1, 1], None, None],
            [[0, 0, 0, 1, 1, 1], None, [0, 1, 2]],
            [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1], [0, 1, 2]],
            [[0, 1, 1, 1, 0, 1], None, [0, 4]],
            [None, [0, 1, 1, 1, 0, 1], [0, 4]],
            [None, [0, 1, 0, 0, 0, 0], None],
            [None, [0, 1, 0, 0, 0, 0], [0, 1, 2]],
        ],
    )
    def test_partials(self, segment, indices, problem):
        con = pcons.ContinuityConstraint(segment, indices=indices)
        partials = con.partials(problem.freeVarIndexMap())

        assert isinstance(partials, dict)

        assert segment.origin.state in partials
        dF_dq0 = partials[segment.origin.state]
        assert isinstance(dF_dq0, np.ndarray)
        assert dF_dq0.shape == (con.size, segment.origin.state.values.size)

        assert segment.terminus.state in partials
        dF_dqf = partials[segment.terminus.state]
        assert isinstance(dF_dqf, np.ndarray)
        assert dF_dqf.shape == (con.size, segment.terminus.state.values.size)

        assert segment.tof in partials
        dF_dtof = partials[segment.tof]
        assert isinstance(dF_dtof, np.ndarray)
        assert dF_dtof.shape == (con.size,)

        assert not segment.propParams in partials
        assert not segment.origin.epoch in partials
        assert not segment.terminus.epoch in partials

        prob = ShootingProblem()
        prob.addSegments(segment)
        prob.addConstraints(con)
        prob.build()
        assert prob.checkJacobian()

    @pytest.mark.parametrize(
        "originMask, terminusMask, indices",
        [
            [None, None, None],
            [None, None, [0, 1, 2]],
            [None, [0, 0, 0, 1, 1, 1], [0, 1, 2]],
            [[0, 0, 0, 1, 1, 1], None, None],
            [[0, 1, 0, 1, 1, 1], None, None],
            [[0, 0, 0, 1, 1, 1], None, [0, 1, 2]],
            [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1], [0, 1, 2]],
        ],
    )
    def test_excludeTerminus(self, segment, indices):
        # Don't include the terminus in the free variable vector
        prob = CorrectionsProblem()
        prob.addVariables(segment.origin.state)
        prob.addVariables(segment.tof)
        con = pcons.ContinuityConstraint(segment, indices=indices)
        err = con.evaluate(prob.freeVarIndexMap())
        assert isinstance(err, np.ndarray)
        assert err.size == con.size

        partials = con.partials(prob.freeVarIndexMap())
        assert not segment.terminus.state in partials
        assert segment.origin.state in partials
        assert segment.tof in partials
        assert not segment.propParams in partials
        assert not segment.origin.epoch in partials
        assert not segment.terminus.epoch in partials

        assert prob.checkJacobian()

    @pytest.mark.parametrize(
        "oMask, tMask, indices",
        [
            [None, None, None],
            [None, None, [0, 1, 2]],
            [None, [0, 0, 0, 1, 1, 1], [0, 1, 2]],
            [[0, 0, 0, 1, 1, 1], None, None],
            [[0, 1, 0, 1, 1, 1], None, None],
            [[0, 0, 0, 1, 1, 1], None, [0, 1, 2]],
            [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1], [0, 1, 2]],
        ],
    )
    def test_paramPartials(self, oMask, tMask, indices):
        from pika.lowthrust.control import (
            ConstMassTerm,
            ConstOrientTerm,
            ConstThrustTerm,
            ForceMassOrientLaw,
        )

        thrust = ConstThrustTerm(7e-2)
        mass = ConstMassTerm(1.0)
        orient = ConstOrientTerm(-76.5 * np.pi / 180.0, 0.0)
        control = ForceMassOrientLaw(thrust, mass, orient)

        from pika.lowthrust.dynamics import LowThrustCrtbpDynamics

        model = LowThrustCrtbpDynamics(loadBody("Earth"), loadBody("Moon"), control)

        q0 = [0.8213, 0.0, 0.5690, 0.0, -1.8214, 0.0]
        state0 = Variable(q0, mask=oMask)
        statef = Variable(q0, mask=tMask)
        origin = ControlPoint(model, 0.0, state0)
        terminus = ControlPoint(model, 0.0, statef)
        segment = Segment(origin, 1.0, terminus, propParams=control.params)
        con = pcons.ContinuityConstraint(segment, indices=indices)

        prob = ShootingProblem()
        prob.addSegments(segment)
        prob.addConstraints(con)
        prob.build()

        partials = con.partials(prob.freeVarIndexMap())
        assert segment.propParams in partials
        dF_dp = partials[segment.propParams]
        assert isinstance(dF_dp, np.ndarray)
        assert dF_dp.shape == (con.size, len(control.params))
        assert not all(x == 0.0 for x in dF_dp.flat)


class TestVariableValueConstraint:
    @pytest.mark.parametrize(
        "var",
        [
            Variable([1.0, 2.0]),
            Variable([1.0, 2.0], [False, True]),
        ],
    )
    @pytest.mark.parametrize("values", [[0.0, 0.0], [None, 0.0], [None, None]])
    def test_constructor(self, var, values):
        con = pcons.VariableValueConstraint(var, values)
        assert con.variable == var

        assert isinstance(con.values, np.ma.MaskedArray)
        assert np.array_equal(con.values, values)
        assert np.array_equal(con.values.mask, [v is None for v in values])

    @pytest.mark.parametrize(
        "var",
        [
            Variable([1.0]),  # incorrect number of elements
            np.array([1.0, 2.0]),  # incorrect type
        ],
    )
    def test_constructor_errors(self, var):
        with pytest.raises(ValueError):
            pcons.VariableValueConstraint(var, [0.0, 0.0])

    @pytest.mark.parametrize(
        "var",
        [
            Variable([1.0, 2.0]),
            Variable([1.0, 2.0], [False, True]),
        ],
    )
    @pytest.mark.parametrize("values", [[0.0, 0.0], [None, 0.0], [None, None]])
    def test_size(self, var, values):
        con = pcons.VariableValueConstraint(var, values)
        assert con.size == sum([v is not None for v in values])

    @pytest.mark.parametrize(
        "values, varMask",
        [
            [[0.0, 0.0], [0, 0]],
            [[None, 0.0], [0, 0]],
            [[None, None], [0, 0]],
            [[0.0, 0.0], [1, 0]],
            [[None, 0.0], [1, 0]],
            [[None, 0.0], [0, 1]],
            [[None, None], [1, 0]],
        ],
    )
    def test_evaluate(self, values, varMask):
        var = Variable([1.0, 2.0], mask=varMask)
        con = pcons.VariableValueConstraint(var, values)
        indexMap = {var: 3}
        conEval = con.evaluate(indexMap)

        assert isinstance(conEval, np.ndarray)
        assert conEval.size == con.size

        count = 0
        for ix in range(len(con.values)):
            if not con.values.mask[ix]:
                assert conEval[count] == var.allVals[ix] - con.values[ix]
                count += 1

    @pytest.mark.parametrize(
        "values, varMask",
        [
            [[0.0, 0.0], [0, 0]],
            [[None, 0.0], [0, 0]],
            [[None, None], [0, 0]],
            [[0.0, 0.0], [1, 0]],
            [[None, 0.0], [1, 0]],
            [[None, 0.0], [0, 1]],
            [[None, None], [1, 0]],
        ],
    )
    def test_partials(self, values, varMask):
        var = Variable([1.0, 2.0], mask=varMask)
        con = pcons.VariableValueConstraint(var, values)
        problem = CorrectionsProblem()
        problem.addVariables(var)
        problem.addConstraints(con)

        partials = con.partials(problem.freeVarIndexMap())

        assert isinstance(partials, dict)
        assert var in partials
        assert len(partials) == 1
        assert partials[var].shape == (con.size, var.values.size)
        assert problem.checkJacobian()

    def test_varsAreRefs(self):
        var = Variable([1.0, 2.0])
        con = pcons.VariableValueConstraint(var, [0.0, 1.0])
        indexMap = {var: 0}

        conEval = con.evaluate(indexMap)
        assert np.array_equal(conEval, [1.0, 1.0])

        # Update variable
        var.values[:] = [0.0, 1.0]
        conEval2 = con.evaluate(indexMap)
        assert np.array_equal(conEval2, [0.0, 0.0])
