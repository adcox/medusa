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

        if not len(indices) == len(np.unique(indices)):
            raise RuntimeError(f"Indices cannot have repeated values")

        self.constrainedIx = sorted(indices)

    @property
    def size(self):
        return len(self.constrainedIx)

    def clearCache(self):
        self.segment.resetProp()

    def evaluate(self, freeVarIndexMap):
        # F = propFinalState - terminalState
        termVar = self.segment.terminus.state
        propState = self.segment.state(-1)[self.constrainedIx]
        termState = termVar.allVals[self.constrainedIx]
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
            dF_dqf = np.zeros((self.size, termVar.values.size))
            for count, ix in enumerate(self.constrainedIx):
                dF_dqf[count, ix] = -1

            partials[termVar] = dF_dqf

        # Partials for other variables come from integrated solution; ask for them
        #   in decreasing order of complexity to take advantage of lazy propagation
        if paramVar in freeVarIndexMap:
            dState_dParams = self.segment.partials_state_wrt_params(-1)
            partials[paramVar] = dState_dParams[self.constrainedIx, :]

        if epochVar in freeVarIndexMap:
            dState_dEpoch = self.segment.partials_state_wrt_epoch(-1)
            partials[epochVar] = dState_dEpoch[self.constrainedIx]

        if originVar in freeVarIndexMap:
            # rows of STM correspond to terminal state, cols to origin state
            stm = self.segment.partials_state_wrt_initialState(-1)
            stm_free = stm[self.constrainedIx, :]
            partials[originVar] = stm_free

        if tofVar in freeVarIndexMap:
            dState_dtof = self.segment.partials_state_wrt_time(-1)
            partials[tofVar] = dState_dtof[self.constrainedIx]

        return partials


class VariableValueConstraint(AbstractConstraint):
    """
    Constrain a variable to have specified values

    Args:
        variable (Variable): the variable to cosntrain
        values (numpy.ndarray of float): values for the variable. The size of
            ``values`` must match the size of the variable regardless of its
            masking. A ``None`` value in ``values`` indicates that the corresponding
            free value in ``variable`` is unconstrained.
    """

    def __init__(self, variable, values):
        if not isinstance(variable, Variable):
            raise ValueError("variable must be a Variable object")

        values = np.array(values, ndmin=1)
        if not values.size == variable.values.size:
            raise ValueError(
                f"Values has {values.size} elements, but must have same number "
                f"as variable ({variable.values.size})"
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

        return self.variable.allVals[~self.values.mask] - self.values[~self.values.mask]

    def partials(self, freeVarIndexMap):
        # Partial is 1 for each constrained variable, zero otherwise
        deriv = np.zeros((self.size, self.variable.values.size))
        count = 0
        for ix, val in enumerate(self.values):
            if not self.values.mask[ix]:
                deriv[count, ix] = 1
                count += 1

        return {self.variable: deriv}
