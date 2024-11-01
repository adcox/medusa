"""
Constraints
============

A variety of constraints are defined and can be applied to problems in different
ways.

.. autosummary::
   Angle
   Inequality
   StateContinuity
   VariableValue

Reference
==============

.. autoclass:: Angle
   :members:

.. autoclass:: Inequality
   :members:

.. autoclass:: StateContinuity
   :members:

.. autoclass:: VariableValue
   :members:

"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from enum import IntEnum
from typing import Union

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

from medusa import util
from medusa.corrections import AbstractConstraint, ControlPoint, Segment, Variable
from medusa.typing import FloatArray, IntArray, override


class Angle(AbstractConstraint):
    """
    Constrain the angle between part of the state vector and a prescribed vector

    TODO document
    """

    def __init__(
        self,
        refDir: FloatArray,
        state: Variable,
        angle: float,
        stateIx: IntArray = (0, 1, 2),
        center: FloatArray = (0.0, 0.0, 0.0),
    ) -> None:
        # TODO document attributes
        self.refDir: NDArray[np.double] = np.array(refDir, copy=True)
        self.state: Variable = state  # TODO check type
        self.angleVal: float = float(np.cos(angle * np.pi / 180.0))
        self.stateIx: NDArray[np.int_] = np.array(stateIx, copy=True)
        self.center: NDArray[np.double] = np.array(center, copy=True)

        if self.refDir.size < 2:
            raise ValueError("refDir must be at least 2D")

        if np.linalg.norm(self.refDir) == 0.0:
            raise ValueError("refDir cannot be zero")

        if not self.refDir.size == self.stateIx.size:
            raise ValueError("refDir must have same number of elements as stateIx")

        if not self.center.size == self.stateIx.size:
            raise ValueError("center must have same number of elements as stateIx")

        # Try evaluating state and indices to raise index error
        self.state.data[self.stateIx]

    def __repr__(self) -> str:
        return util.repr(self, "refDir", "state", "angle", "stateIx", "center")

    @property
    @override
    def size(self) -> int:
        return 1

    @override
    def evaluate(self) -> NDArray[np.double]:
        # TODO note about no sign distinction due to cosine?
        # Use dot product to compute angle between vectors
        stateVec = self.state.data[self.stateIx] - self.center
        evalAngle = float(self.refDir.T @ stateVec)
        return np.asarray(evalAngle - self.angleVal)

    @override
    def partials(
        self, freeVarIndexMap: dict[Variable, int]
    ) -> dict[Variable, NDArray[np.double]]:
        if self.state in freeVarIndexMap:
            dFdq = np.zeros((1, self.state.size))
            dFdq[0, self.stateIx] = self.refDir
            return {self.state: dFdq}
        else:
            return {}


class StateContinuity(AbstractConstraint):
    """
    Constrain the terminal end of a segment to match the control point state

    TODO document
    """

    def __init__(self, segment: Segment, indices: Union[None, IntArray] = None) -> None:
        if segment.terminus is None:
            raise RuntimeError("Cannot constraint StateContinuity with terminus = None")

        self.segment = segment

        if indices is None:
            # default to constraining all state variables
            indices = np.arange(len(segment.terminus.state.data))

        if not len(indices) == len(np.unique(indices)):
            raise RuntimeError("Indices cannot have repeated values")

        self.constrainedIx = sorted(indices)

    def __repr__(self) -> str:
        return util.repr(self, "segment", "constrainedIx")

    @property
    @override
    def size(self) -> int:
        return len(self.constrainedIx)

    @override
    def clearCache(self) -> None:
        self.segment.resetProp()

    @override
    def evaluate(self) -> NDArray[np.double]:
        # F = propFinalState - terminalState

        # Already asserted that segment.terminus is not None
        termVar = self.segment.terminus.state  # type: ignore
        propState = self.segment.state(-1)[self.constrainedIx]
        termState = termVar.data[self.constrainedIx]
        return propState - termState

    @override
    def partials(
        self, freeVarIndexMap: dict[Variable, int]
    ) -> dict[Variable, NDArray[np.double]]:
        originVar = self.segment.origin.state
        epochVar = self.segment.origin.epoch
        termVar = self.segment.terminus.state  # type: ignore
        tofVar = self.segment.tof
        paramVar = self.segment.propParams

        partials = {}
        if termVar in freeVarIndexMap:
            # Partials for terminal state are all -1
            dF_dqf = np.zeros((self.size, termVar.size))
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

    def __init__(self, variable: Variable, values: FloatArray) -> None:
        if not isinstance(variable, Variable):
            raise ValueError("variable must be a Variable object")

        values = np.array(values, ndmin=1, copy=True)
        if not values.size == variable.size:
            raise ValueError(
                f"Values has {values.size} elements, but must have same number "
                f"as variable ({variable.size})"
            )

        self.variable = variable
        self.values = np.ma.array(values, mask=[v is None for v in values])

    def __repr__(self) -> str:
        return util.repr(self, "values", "variable")

    @property
    @override
    def size(self) -> int:
        return sum(~self.values.mask)

    @override
    def evaluate(self) -> NDArray[np.double]:
        return self.variable.data[~self.values.mask] - self.values[~self.values.mask]

    @override
    def partials(
        self, freeVarIndexMap: dict[Variable, int]
    ) -> dict[Variable, NDArray[np.double]]:
        # Partial is 1 for each constrained variable, zero otherwise
        deriv = np.zeros((self.size, self.variable.size))
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
        constraint: the equality constraint
        mode: the inequality mode
        defaultSlackVal: the default slack variable value if it cannot be
            computed during instantiation
    """

    def __init__(
        self,
        constraint: AbstractConstraint,
        mode: Inequality.Mode,
        defaultSlackValue: float = 1e-6,
    ) -> None:
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

    def __repr__(self) -> str:
        return util.repr(self, "equalCon", "mode", "slack")

    @property
    @override
    def size(self) -> int:
        return self.equalCon.size

    @property
    def importableVars(self) -> list[Variable]:
        """
        A list of variables that can be imported into a corrections problem.

        Importing is accomplished via :func:`CorrectionsProblem.importVariables`.
        These variables are defined as a property so that evaluation occurs at
        runtime, allowing the user to modify or reassign the variables.
        """
        impVars = util.toList(getattr(self.equalCon, "importableVars", []))
        impVars.append(self.slack)
        return impVars

    @override
    def clearCache(self) -> None:
        self.equalCon.clearCache()
        self._slackInit = False  # Force re-init slack variable values

    @override
    def evaluate(self) -> NDArray[np.double]:
        vals = self.equalCon.evaluate()
        return np.asarray(
            [
                val - float(self.mode) * slack * slack
                for val, slack in zip(vals, self.slack)
            ]
        )

    @override
    def partials(
        self, freeVarIndexMap: dict[Variable, int]
    ) -> dict[Variable, NDArray[np.double]]:
        # Compute partials of equality constraint
        partials = self.equalCon.partials(freeVarIndexMap)

        # Append partials specific to inequality: the slack variable(s)
        deriv = np.zeros((self.size, self.slack.size))
        count = 0
        for ix, val in enumerate(self.slack.data):
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
