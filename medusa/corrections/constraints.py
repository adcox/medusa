"""
Core Corrections Class
"""
import logging
from enum import IntEnum

import numpy as np

logger = logging.getLogger(__name__)

from medusa import util
from medusa.corrections import AbstractConstraint, Variable


class StateContinuity(AbstractConstraint):
    """
    Constrain the terminal end of a segment to match the control point state
    """

    def __init__(self, segment, indices=None):
        if segment.terminus is None:
            raise RuntimeError("Cannot constraint StateContinuity with terminus = None")

        self.segment = segment

        if indices is None:
            # default to constraining all state variables
            indices = np.arange(len(segment.terminus.state.allVals))

        if not len(indices) == len(np.unique(indices)):
            raise RuntimeError("Indices cannot have repeated values")

        self.constrainedIx = sorted(indices)

    @property
    def size(self):
        return len(self.constrainedIx)

    def clearCache(self):
        self.segment.resetProp()

    def evaluate(self):
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


class VariableValue(AbstractConstraint):
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

    def evaluate(self):
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


class Inequality(AbstractConstraint):
    """
    Convert an equality constraint into an inequality constraint.

    Args:
        constraint (AbstractConstraint): the equality constraint
        mode (Inequality.Mode): the inequality mode
        defaultSlackVal(Optional, float): the default slack variable value if it
            cannot be computed during instantiation
    """

    def __init__(self, constraint, mode, defaultSlackValue=1e-6):
        if not isinstance(constraint, AbstractConstraint):
            raise TypeError("constraint must be derived from AbstractConstraint")
        if not isinstance(mode, Inequality.Mode):
            raise TypeError("mode must be Inequality.Mode")

        self.equalCon = constraint
        self.mode = mode

        # Define slack variable(s)
        slackVals = [defaultSlackValue] * constraint.size
        # Try to compute slack variable values such that the constraint is
        #   initially satisfied
        try:
            F_equal = self.equalCon.evaluate()
            sign = -1 * self.mode
            for ix, F in enumerate(F_equal):
                # Set the slack variable value if it is real-valued
                if F * sign <= 0:
                    slackVals[ix] = np.sqrt(-F * sign)
        except Exception:
            logger.debug("Could not update initial slack variable values")

        self.slack = Variable(slackVals, name="Slack")

    @property
    def size(self):
        return self.equalCon.size

    @property
    def importableVars(self):
        impVars = util.toList(getattr(self.equalCon, "importableVars", []))
        impVars.append(self.slack)
        return impVars

    def clearCache(self):
        self.equalCon.clearCache()
        self._slackInit = False  # Force re-init slack variable values

    def evaluate(self):
        vals = self.equalCon.evaluate()
        return np.asarray(
            [
                val - float(self.mode) * slack * slack
                for val, slack in zip(vals, self.slack.values)
            ]
        )

    def partials(self, freeVarIndexMap):
        # Compute partials of equality constraint
        partials = self.equalCon.partials(freeVarIndexMap)

        # Append partials specific to inequality: the slack variable(s)
        deriv = np.zeros((self.size, self.slack.values.size))
        count = 0
        for ix, val in enumerate(self.slack.allVals):
            if not self.slack.mask[ix]:
                deriv[count, ix] = -2.0 * self.mode * val
                count += 1

        partials[self.slack] = deriv
        return partials

    class Mode(IntEnum):
        LESS = -1
        """
        Less-than
        """

        GREATER = 1
        """
        Greater-than
        """
