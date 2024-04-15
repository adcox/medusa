"""
Test Constraint objects
"""
import numpy as np
import pytest

import pika.corrections.constraints as pcons
from pika.corrections import Variable


class TestVariableValueConstraint:
    @pytest.mark.parametrize(
        "var",
        [
            Variable([1.0, 2.0]),
            Variable([1.0, 2.0, 3.0], [False, False, True]),
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
            Variable([1.0, 2.0], [True, False]),  # incorrect number of free vars
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
            Variable([1.0, 2.0, 3.0], [False, False, True]),
        ],
    )
    @pytest.mark.parametrize("values", [[0.0, 0.0], [None, 0.0], [None, None]])
    def test_size(self, var, values):
        con = pcons.VariableValueConstraint(var, values)
        assert con.size == sum([v is not None for v in values])

    @pytest.mark.parametrize("values", [[0.0, 0.0], [None, 0.0], [None, None]])
    def test_evaluate(self, values):
        var = Variable([1.0, 2.0])
        con = pcons.VariableValueConstraint(var, values)
        indexMap = {var: 3}
        freeVarVec = np.array([0.0, 1.0, 2.0, 1.1, 2.05])
        conEval = con.evaluate(indexMap, freeVarVec)

        assert isinstance(conEval, np.ndarray)
        assert conEval.size == con.size

        # Constraint eval should use free variable values from the vector, NOT
        #   from the object used to construct the constraint
        count = 0
        for ix in range(len(con.values)):
            if not con.values.mask[ix]:
                assert conEval[count] == freeVarVec[indexMap[var] + ix] - con.values[ix]
                count += 1

    @pytest.mark.parametrize("values", [[0.0, 0.0], [None, 0.0], [None, None]])
    def test_partials(self, values):
        var = Variable([1.0, 2.0])
        con = pcons.VariableValueConstraint(var, values)
        indexMap = {var: 3}
        freeVarVec = np.array([0.0, 1.0, 2.0, 1.1, 2.05])
        partials = con.partials(indexMap, freeVarVec)

        assert isinstance(partials, dict)
        assert var in partials
        assert len(partials) == 1

        assert partials[var].shape == (con.size, var.numFree)

        count = 0
        for ix in range(len(con.values)):
            if not con.values.mask[ix]:
                assert partials[var][count, ix] == 1
                count += 1
