"""
Test corrections
"""
import numpy as np
import pytest

import pika.corrections.constraints as constraints
from pika.corrections import AbstractConstraint, CorrectionsProblem, Variable


@pytest.mark.parametrize(
    "vals, mask, name",
    [
        [1.0, False, "state"],
        [[1.0, 2.0], [True, False], "angle"],
    ],
)
def test_Variable(vals, mask, name):
    var = Variable(vals, mask, name)

    assert isinstance(var.values, np.ndarray)
    assert len(var.values.shape) == 1

    assert all(var.values.data == np.array(vals, ndmin=1))
    assert all(var.values.mask == np.array(mask, ndmin=1))
    assert var.name == name


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
