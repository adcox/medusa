"""
Core Corrections Class
"""
import logging

import numpy as np

logger = logging.getLogger(__name__)

from pika.corrections import AbstractConstraint, Variable


class ContinuityConstraint(AbstractConstraint):
    """
    Constrain the end of a segment to match a control point
    """

    def __init__(self, segment, indices=None):
        if segment.terminus is None:
            raise RuntimeError("Cannot constraint continuity with terminus = None")

        self.segment = segment

        if indices is None:
            # default to constraining all state variables
            indices = np.arange(len(segment.terminus.state.allVals))

        # TODO does this need to be set to None initially and computed later?
        self.unmaskedIx = segment.terminus.state.unmaskedIndices(indices)
        if not len(self.unmaskedIx) == len(indices):
            raise RuntimeError(
                f"ContinuityConstraint cannot be applied to ix = {indices};"
                f" only {len(self.unmaskedIx)} of these indices are free variables"
            )

    @property
    def size(self):
        return len(self.unmaskedIx)

    def clearCache(self):
        self.segment.resetProp()

    def evaluate(self, freeVarIndexMap):
        # F = propFinalState - terminalState
        termVar = self.segment.terminus.state
        propState = self.segment.finalState()[~termVar.mask][self.unmaskedIx]
        termState = termVar.freeVals[self.unmaskedIx]
        return propState - termState

    def partials(self, freeVarIndexMap):
        originVar = self.segment.origin.state
        epochVar = self.segment.origin.epoch
        termVar = self.segment.terminus.state
        tofVar = self.segment.tof
        paramVar = self.segment.propParams

        partials = {}
        if termVar in freeVarIndexMap:
            # Partials for terminal state are all -1
            dF_dqf = np.zeros((self.size, termVar.numFree))
            for count, ix in enumerate(self.unmaskedIx):
                dF_dqf[count, ix] = -1
            partials[termVar] = dF_dqf

        # Partials for other variables come from integrated solution; ask for them
        #   in decreasing order of complexity to take advantage of lazy propagation
        if paramVar in freeVarIndexMap:
            dState_dParams = self.segment.partials_finalState_wrt_params()
            partials[paramVar] = dState_dParams[:, ~termVar.mask][:, self.unmaskedIx][
                ~paramVar.mask, :
            ]

        if epochVar in freeVarIndexMap:
            dState_dEpoch = self.segment.partials_finalState_wrt_epoch()
            partials[epochVar] = dState_dEpoch[~termVar.mask][self.unmaskedIx]

        if originVar in freeVarIndexMap:
            # rows of STM correspond to terminal state, cols to origin state
            stm = self.segment.partials_finalState_wrt_initialState()
            stm_free = stm[~termVar.mask, :][self.unmaskedIx, :][:, ~originVar.mask]
            partials[originVar] = stm_free

        if tofVar in freeVarIndexMap:
            dState_dtof = self.segment.partials_finalState_wrt_time()
            partials[tofVar] = dState_dtof[~termVar.mask][self.unmaskedIx]

        return partials


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

        self.variable = variable
        self.values = np.ma.array(values, mask=[v is None for v in values])

    @property
    def size(self):
        return sum(~self.values.mask)

    def evaluate(self, freeVarIndexMap):
        if not self.variable in freeVarIndexMap:
            # TODO handle more gracefully?
            raise RuntimeError(f"{self.variable} is not in free variable index map")

        return (
            self.variable.freeVals[~self.values.mask] - self.values[~self.values.mask]
        )
        # ix0 = freeVarIndexMap[self.variable]
        # vecValues = freeVarVec[ix0 : ix0 + self.variable.numFree]
        # return vecValues[~self.values.mask] - self.values[~self.values.mask]

    def partials(self, freeVarIndexMap):
        # Partial is 1 for each constrained variable, zero otherwise
        deriv = np.zeros((self.size, self.variable.numFree))
        count = 0
        for ix, val in enumerate(self.values):
            if not self.values.mask[ix]:
                deriv[count, ix] = 1
                count += 1

        return {self.variable: deriv}
