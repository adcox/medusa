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
    def test_constructor(self, var):
        values = [0.0, 0.0]
        con = pcons.VariableValueConstraint(var, values)
        assert all([cval == val for cval, val in zip(con.values, values)])
        assert con.variable == var

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
    def test_size(self, var):
        con = pcons.VariableValueConstraint(var, [0.0, 0.0])
        assert con.size == 2

    def test_evaluate(self):
        var = Variable([1.0, 2.0])
        con = pcons.VariableValueConstraint(var, [0.0, 0.0])
        indexMap = {var: 3}
        freeVarVec = np.array([0.0, 1.0, 2.0, 3.0, 5.0])
        conVal = con.evaluate(indexMap, freeVarVec)

        assert isinstance(conVal, np.ndarray)
        assert conVal.size == con.size

        # Constraint eval should use free variable values from the vector, NOT
        #   from the object used to construct the constraint
        assert all(
            [
                conVal[ix] == freeVarVec[indexMap[var] + ix] - con.values[ix]
                for ix in range(con.size)
            ]
        )

    # TODO test partials
