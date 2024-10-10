"""
Test Constraint objects
"""
import numpy as np
import pytest
from conftest import loadBody

import medusa.corrections.constraints as pcons
from medusa.corrections import (
    ControlPoint,
    CorrectionsProblem,
    DifferentialCorrector,
    Segment,
    ShootingProblem,
    Variable,
)
from medusa.dynamics.crtbp import DynamicsModel as crtbpModel

emModel = crtbpModel(loadBody("Earth"), loadBody("Moon"))


# ------------------------------------------------------------------------------
# Fixtures to create a simple corrections problem
# ------------------------------------------------------------------------------
@pytest.fixture
def origin(originMask):
    # IC for EM L3 Vertical
    state = Variable([0.8213, 0.0, 0.5690, 0.0, -1.8214, 0.0], originMask)
    return ControlPoint(emModel, 0.1, state)


@pytest.fixture
def terminus(terminusMask):
    state = Variable([0.0] * 6, terminusMask)
    return ControlPoint(emModel, 1.1, state)


@pytest.fixture
def segment(origin, terminus):
    return Segment(origin, 1.0, terminus=terminus)


@pytest.fixture
def problem(segment):
    prob = ShootingProblem()
    prob.addSegments(segment)
    prob.build()

    return prob


class TestAngle:
    @pytest.mark.parametrize("originMask", [None])
    @pytest.mark.parametrize(
        "refDir, stateIx, center",
        [
            [(0, 0, 1), (0, 1, 2), (0, 0, 0)],  # position angle
            [(0, 0, 1), (3, 4, 5), (0, 0, 0)],  # velocity angle
            [(1, 0), (0, 1), (0, 0)],  # 2D
            [(1, 0, 0, 0), (1, 2, 3, 4), (0, -1, 1, 0)],  # 4D
        ],
    )
    def test_constructor(self, origin, refDir, stateIx, center):
        angle = 60
        con = pcons.Angle(refDir, origin.state, angle, stateIx, center)

        assert isinstance(con.refDir, np.ndarray)
        np.testing.assert_array_equal(con.refDir, refDir)

        assert isinstance(con.stateIx, np.ndarray)
        np.testing.assert_array_equal(con.stateIx, stateIx)

        assert isinstance(con.center, np.ndarray)
        np.testing.assert_array_equal(con.center, center)

        assert id(origin.state) == id(con.state)

    @pytest.mark.parametrize("originMask", [None])
    @pytest.mark.parametrize(
        "refDir, stateIx, center, err",
        [
            [(1, 0, 0), (0, 1, 2), (0, 0), ValueError],  # invalid center size
            [
                (1, 0, 0),
                (0, 1, 2, 3, 4, 5),
                (0, 0, 0),
                ValueError,
            ],  # invalid stateIx size
            [(0, 1), (0, 1, 2), (0, 0, 0), ValueError],  # invalid refDir size
            [(0, 0), (0, 1), (0, 0), ValueError],  # refDir is zero
            [(0, 1, 2), (0, 1, 6), (0, 0, 0), IndexError],  # index out of bounds
        ],
    )
    def test_constructor_err(self, origin, refDir, stateIx, center, err):
        with pytest.raises(err):
            pcons.Angle(refDir, origin.state, 30, stateIx, center)

    @pytest.mark.parametrize("originMask", [None])
    @pytest.mark.parametrize(
        "refDir, stateIx, center",
        [
            [(1, 0, 0), (0, 1, 2), (1, 0, 0)],  # position angle
            [(0, 0, 1), (3, 4, 5), (0, 0, 0)],  # velocity angle
            [(1, 0), (0, 1), (0, 0)],  # 2D
            [(1, 0, 0, 0), (1, 2, 3, 4), (0, 0, 0, 0)],  # 4D
        ],
    )
    def test_evaluate(self, origin, refDir, stateIx, center):
        angle = 27.0
        con = pcons.Angle(refDir, origin.state, angle, stateIx, center)
        out = con.evaluate()
        assert isinstance(out, np.ndarray)

    @pytest.mark.parametrize("originMask, terminusMask", [[None, None]])
    @pytest.mark.parametrize(
        "refDir, stateIx, center",
        [
            [(1, 0.1, 0.2), (0, 1, 2), (1, 0, 0)],  # position angle
            [(0, 0, 1), (3, 4, 5), (0, 0, 0)],  # velocity angle
            [(1, 0), (0, 1), (0, 0)],  # 2D
            [(1, 0, 0, 0), (1, 2, 3, 4), (0, 0, 0, 0)],  # 4D
        ],
    )
    def test_partials(self, origin, refDir, stateIx, center, problem):
        angle = 27.0
        con = pcons.Angle(refDir, origin.state, angle, stateIx, center)
        problem.addConstraints(con)
        assert problem.checkJacobian()


class TestStateContinuity:
    @pytest.mark.parametrize(
        "originMask, terminusMask, indices",
        [
            [None, None, None],
            [None, None, [0, 1, 2]],
            [None, [0, 0, 0, 1, 1, 1], [0, 1, 2]],
        ],
    )
    def test_constructor(self, segment, indices):
        con = pcons.StateContinuity(segment, indices=indices)

    @pytest.mark.parametrize(
        "originMask, terminusMask, indices",
        [
            [None, [0, 1, 0, 0, 0, 0], [0, 1, 1]],
        ],
    )
    def test_constructor_err(self, segment, indices):
        with pytest.raises(RuntimeError):
            pcons.StateContinuity(segment, indices=indices)

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
        con = pcons.StateContinuity(segment, indices=indices)
        err = con.evaluate()
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
        con = pcons.StateContinuity(segment, indices=indices)
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
        con = pcons.StateContinuity(segment, indices=indices)
        err = con.evaluate()
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
        from medusa.dynamics.lowthrust import (
            ConstMassTerm,
            ConstOrientTerm,
            ConstThrustTerm,
            ForceMassOrientLaw,
        )

        thrust = ConstThrustTerm(7e-2)
        mass = ConstMassTerm(1.0)
        orient = ConstOrientTerm(-76.5 * np.pi / 180.0, 0.0)
        control = ForceMassOrientLaw(thrust, mass, orient)

        from medusa.dynamics.lowthrust.crtbp import DynamicsModel as LTDynamics

        model = LTDynamics(loadBody("Earth"), loadBody("Moon"), control)

        q0 = [0.8213, 0.0, 0.5690, 0.0, -1.8214, 0.0]
        state0 = Variable(q0, mask=oMask)
        statef = Variable(q0, mask=tMask)
        origin = ControlPoint(model, 0.0, state0)
        terminus = ControlPoint(model, 0.0, statef)
        segment = Segment(origin, 1.0, terminus, propParams=control.params)
        con = pcons.StateContinuity(segment, indices=indices)

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

        assert prob.checkJacobian()


class TestVariableValue:
    @pytest.mark.parametrize(
        "var",
        [
            Variable([1.0, 2.0]),
            Variable([1.0, 2.0], [False, True]),
        ],
    )
    @pytest.mark.parametrize("values", [[0.0, 0.0], [None, 0.0], [None, None]])
    def test_constructor(self, var, values):
        con = pcons.VariableValue(var, values)
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
            pcons.VariableValue(var, [0.0, 0.0])

    @pytest.mark.parametrize(
        "var",
        [
            Variable([1.0, 2.0]),
            Variable([1.0, 2.0], [False, True]),
        ],
    )
    @pytest.mark.parametrize("values", [[0.0, 0.0], [None, 0.0], [None, None]])
    def test_size(self, var, values):
        con = pcons.VariableValue(var, values)
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
        con = pcons.VariableValue(var, values)
        conEval = con.evaluate()

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
        con = pcons.VariableValue(var, values)
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
        con = pcons.VariableValue(var, [0.0, 1.0])

        conEval = con.evaluate()
        assert np.array_equal(conEval, [1.0, 1.0])

        # Update variable
        var.values[:] = [0.0, 1.0]
        conEval2 = con.evaluate()
        assert np.array_equal(conEval2, [0.0, 0.0])


class TestInequalityConstraint:
    @pytest.mark.parametrize(
        "mode", [pcons.Inequality.Mode.LESS, pcons.Inequality.Mode.GREATER]
    )
    def test_constructor(self, mode):
        var = Variable([1.0, 2.0])
        values = [1.1, 1.8]
        eqCon = pcons.VariableValue(var, values)
        ineqCon = pcons.Inequality(eqCon, mode, defaultSlackValue=1e-3)

        assert id(ineqCon.equalCon) == id(eqCon)
        assert ineqCon.mode == mode
        assert isinstance(ineqCon.importableVars, list)
        assert len(ineqCon.importableVars) == 1
        assert ineqCon.importableVars[0] == ineqCon.slack
        assert ineqCon.slack.values.size == len(values)

        # Check that slack values have been intelligently selected
        if mode < 0:  # less than
            assert ineqCon.slack.values[0] == pytest.approx(np.sqrt(0.1), 1e-6)
            assert ineqCon.slack.values[1] == 1e-3  # default value
        else:
            assert ineqCon.slack.values[0] == 1e-3  # default value
            assert ineqCon.slack.values[1] == pytest.approx(np.sqrt(0.2), 1e-6)

    @pytest.mark.parametrize(
        "mode", [pcons.Inequality.Mode.LESS, pcons.Inequality.Mode.GREATER]
    )
    def test_evaluate(self, mode):
        var = Variable([1.0, 2.0])
        values = [1.1, 1.8]
        eqCon = pcons.VariableValue(var, values)
        ineqCon = pcons.Inequality(eqCon, mode)

        conVals = ineqCon.evaluate()
        assert isinstance(conVals, np.ndarray)
        assert conVals.size == len(values)

    @pytest.mark.parametrize(
        "mode", [pcons.Inequality.Mode.LESS, pcons.Inequality.Mode.GREATER]
    )
    def test_partials(self, mode):
        var = Variable([1.0, 2.0])
        values = [1.1, 1.8]
        eqCon = pcons.VariableValue(var, values)
        ineqCon = pcons.Inequality(eqCon, mode)
        indexMap = {var: 0}

        partials = ineqCon.partials(indexMap)
        assert isinstance(partials, dict)
        assert var in partials
        assert ineqCon.slack in partials
        assert len(list(partials.keys())) == 2

    @pytest.mark.parametrize("originMask, terminusMask", [[None, None]])
    @pytest.mark.parametrize(
        "mode", [pcons.Inequality.Mode.LESS, pcons.Inequality.Mode.GREATER]
    )
    def test_inProblem(self, segment, problem, mode):
        # baseline TOF is 1.0
        eqCon = pcons.VariableValue(segment.tof, [2.0])
        ineqCon = pcons.Inequality(eqCon, mode, defaultSlackValue=1e-3)
        problem.addConstraints(ineqCon)

        solver = DifferentialCorrector()
        solution, log = solver.solve(problem)

        assert log["status"] == "converged"

        tof = solution._segments[0].tof.values[0]
        if mode > 0:  # greater-than
            assert len(log["iterations"]) > 1
            assert tof > eqCon.values[0]
        else:
            # Is satisfied without any iterations
            assert len(log["iterations"]) == 1
            assert tof < eqCon.values[0]

        assert problem.checkJacobian()
