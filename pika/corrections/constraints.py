"""
Core Corrections Class
"""
import logging

import numpy as np

logger = logging.getLogger(__name__)

from .corrections import AbstractConstraint, Variable


class VariableValueConstraint(AbstractConstraint):
    """
    Constrain a variable to have specified values

    Args:
        variable (Variable): the variable to cosntrain
        values (numpy.ndarray<float>): values for the variable. The size of
            ``values`` must match free values in ``variable``, i.e., excluding
            any masked elements. A ``None`` value in ``values`` indicates that
            the corresponding free value in ``variable`` is unconstrained.
    """

    def __init__(self, variable, values):
        if not isinstance(variable, Variable):
            raise ValueError("variable must be a Variable object")

        values = np.array(values, ndmin=1)
        if not values.size == variable.numFree:
            raise ValueError(
                f"Values has {values.size} elements, but must have same number "
                f"as variable free values ({variable.numFree})"
            )

        self.values = np.ma.array(values, mask=[v is None for v in values])
        self.variable = variable

    @property
    def size(self):
        return sum(~self.values.mask)

    def evaluate(self, freeVarIndexMap, freeVarVec):
        # Find values in freeVarVec
        if not self.variable in freeVarIndexMap:
            # TODO handle more gracefully?
            raise RuntimeError(f"{self.variable} is not in free variable index map")

        ix0 = freeVarIndexMap[self.variable]
        vecValues = freeVarVec[ix0 : ix0 + self.variable.numFree]
        return vecValues[~self.values.mask] - self.values[~self.values.mask]

    def partials(self, freeVarIndexMap, freeVarVec):
        # Partial is 1 for each constrained variable, zero otherwise
        deriv = np.zeros((self.size, self.variable.numFree))
        count = 0
        for ix, val in enumerate(self.values):
            if not self.values.mask[ix]:
                deriv[count, ix] = 1
                count += 1

        return {self.variable: deriv}
