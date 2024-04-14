"""
Core Corrections Class
"""
import logging
from abc import ABC, abstractmethod
from copy import copy

import numpy as np
import numpy.ma as ma

from pika.dynamics import AbstractDynamicsModel, EOMVars
from pika.propagate import Propagator

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Basic Building Blocks
#
#   - Variable
#   - AbstractConstraint
# ------------------------------------------------------------------------------


class Variable:
    """
    Contains a variable vector with an optional mask to flag non-variable values

    Args:
        values (float, [float], np.ndarray<float>): scalar or array of variable
            values
        mask (bool, [bool], np.ndarray<bool>): ``True`` flags values as excluded
            from the free variable vector; ``False`` flags values as included
    """

    def __init__(self, values, mask=False, name=""):
        self.values = ma.array(values, mask=mask, ndmin=1)
        self.name = name

    @property
    def allVals(self):
        return self.values.data

    @property
    def freeVals(self):
        return self.values[~self.values.mask]

    @property
    def numFree(self):
        """
        Get the number of un-masked values, i.e., the number of free variables
        within the vector

        Returns:
            int: the number of un-masked values
        """
        return int(sum(~self.values.mask))


class AbstractConstraint(ABC):
    """
    Defines the interface for a constraint object
    """

    @property
    @abstractmethod
    def size(self):
        """
        Get the number of constraint rows, i.e., the number of scalar constraint
        equations

        Returns:
            int: the number of constraint rows
        """
        pass

    @abstractmethod
    def evaluate(self, freeVarIndexMap, freeVarVec):
        """
        Evaluate the constraint

        Args:
            freeVarIndexMap (dict): maps the first index (:class:`int`) of each⋅
                :class:`Variable` within ``freeVars`` to the variable object
            freeVarVec (numpy.ndarray<float>): free variable vector

        Returns:
            numpy.ndarray<float> the value of the constraint funection; evaluates
            to zero when the constraint is satisfied

        Raises:
            RuntimeError: if the constraint cannot be evaluated
        """
        pass

    @abstractmethod
    def partials(self, freeVarIndexMap, freeVarVec):
        """
        Compute the partial derivatives of the constraint vector with respect to
        all variables

        Args:
            freeVarIndexMap (dict): maps the first index (:class:`int`) of each⋅
                :class:`Variable` within ``freeVars`` to the variable object
            freeVarVec (numpy.ndarray<float>): free variable vector

        Returns:
            dict: a dictionary mapping a :class:`Variable` object to the partial
            derivatives of this constraint with respect to that variable⋅
            (:class:`numpy.ndarray<float>`). The partial derivatives with respect
            to variables that are not included in the returned dict are assumed
            to be zero.
        """
        pass


# ------------------------------------------------------------------------------
# Representing Trajectories
#
#   - ControlPoint
#   - Segment
# ------------------------------------------------------------------------------


class ControlPoint:
    """
    Defines a propagation start point

    Args:
        model (AbstractDynamicsModel): defines the dynamics model for the propagation
        epoch (float, Variable): the epoch at which the propagation begins. An
            input ``float`` is converted to a :class:`Variable` with name "Epoch"
        state ([float], Variable): the state at which the propagation begins. An
            input list of floats is converted to a :class:`Variable` with name "State"

    Raises:
        TypeError: if the model is not derived from AbstractDynamicsModel
        RuntimeError: if the epoch specifies more than one value
        RuntimeError: if the state specifies more values than the dynamics model
            allows for :attr:`EOMVars.STATE`
    """

    def __init__(self, model, epoch, state):
        if not isinstance(model, AbstractDynamicsModel):
            raise TypeError("Model must be derived from AbstractDynamicsModel")

        if not isinstance(epoch, Variable):
            epoch = Variable(epoch, name="Epoch")

        if not isinstance(state, Variable):
            state = Variable(state, name="State")

        if not epoch.values.size == 1:
            raise RuntimeError("Epoch can only have one value")

        sz = model.stateSize(EOMVars.STATE)
        if not state.values.size == sz:
            raise RuntimeError("State must have {sz} values")

        self.model = model
        self.epoch = epoch
        self.state = state

    @staticmethod
    def fromProp(solution, ix=0):
        """
        Construct a control point from a propagated arc

        Args:
            solution (scipy.integrate.OptimizeResult): the output from the propagation
            ix (Optional, int): the index of the point within the ``solution``

        Returns:
            ControlPoint: a control point with epoch and state retrieved from the
            propagation solution
        """
        if not isinstance(solution, scipy.integrate.OptimizeResult):
            raise TypeError("Expecting OptimizeResult from scipy solve_ivp")

        if ix > len(solution.t):
            raise ValueError(f"ix = {ix} is out of bounds (max = {len(solution.t)})")

        return ControlPoint(solution.model, solution.t[ix], solution.y[:, ix])


class Segment:
    def __init__(self, origin, tof, prop=None, propParams=[]):
        if not isinstance(origin, ControlPoint):
            raise TypeError("origin must be a ControlPoint")

        if prop is not None:
            if not isinstance(prop, Propagator):
                raise TypeError("prop must be a Propagator")
            prop = copy(prop)
            if not prop.model == origin.model:
                logger.warning("Changing propagator model to match origin")
                prop.model = origin.model
        else:
            prop = Propagator(origin.model, dense=False)

        if not isinstance(tof, Variable):
            tof = Variable(tof, name="Time-of-flight")

        if not isinstance(propParams, Variable):
            propParams = Variable(propParams, name="Params")

        self.origin = origin
        self.tof = tof
        self.prop = prop
        self.propParams = propParams
        self.propSol = None

    def finalState(self):
        self.propagate(EOMVars.STATE)
        return self.propSol.y[:, -1]

    def partials_finalState_wrt_time(self):
        self.propagate(EOMVars.STATE)
        return self.origin.model.evalEOMs(
            self.propSol.t[-1],
            self.propSol.y[:, -1],
            [EOMVars.STATE],
            self.propParams.values,
        )

    def partials_finalState_wrt_initialState(self):
        # Assumes propagation begins at initial state
        self.propagate([EOMVars.STATE, EOMVars.STM])
        return self.origin.model.extractVars(self.propSol.y[:, -1], EOMVars.STM)

    def partials_finalState_wrt_epoch(self):
        self.propagate([EOMVars.STATE, EOMVars.STM, EOMVars.EPOCH_DEPS])
        return self.origin.model.extractVars(self.propSol.y[:, -1], EOMVars.EPOCH_DEPS)

    def partials_finalState_wrt_params(self):
        self.propagate(
            [EOMVars.STATE, EOMVars.STM, EOMVars.EPOCH_DEPS, EOMVars.PARAM_DEPS]
        )
        return self.origin.model.extractVars(self.propSol.y[:, -1], EOMVars.PARAM_DEPS)

    def propagate(self, eomVars, lazy=True):
        # Check to see if we can skip the propagation
        if lazy and self.propSol is not None:
            eomVars = np.array(eomVars, ndmin=1)
            if all([v in self.propSol.eomVars for v in eomVars]):
                return

        # Propagate from the origin for TOF, set self.propSol
        tspan = [0, self.tof.allVals[0]] + self.origin.epoch.allVals[0]
        self.propSol = self.prop.propagate(
            self.origin.state.allVals,
            tspan,
            params=self.propParams.allVals,
            eomVars=eomVars,
        )


# ------------------------------------------------------------------------------
# Organizing it all together
# ------------------------------------------------------------------------------


class CorrectionsProblem:
    """
    Defines a mathematical problem to be solved by a corrections algorithm

    Attributes:
        freeVarIndexMap (dict): maps the first index (:class:`int`) of a variable
            within ``freeVarVec`` to the corresponding :class:`Variable` object.
        constraintIndexMap (dict): maps the first index (:class:`int`) of the
            constraint equation(s) in ``constraintVec`` to the corresponding
            :class:`AbstractConstraint` object.
        freeVarVec (numpy.ndarray<float>): N-element free variable vector
        constraintVec (numpy.ndarray<float>): M-element constraint vector
        jacobian (numpy.ndarray<float>): MxN Jacobian matrix. Each row contains
            the partial derivatives of a constraint equation with respect to
            the free variable vector. Thus, rows correspond to constraints and
            columns correspond to free variables.
    """

    def __init__(self):
        self._freeVarIndexMap = {}
        self._constraintIndexMap = {}

        self._freeVarVec = np.empty((0,))
        self._constraintVec = np.empty((0,))
        self._jacobian = np.empty((0,))

    # -------------------------------------------
    # Variables

    def addVariable(self, variable):
        """
        Add a variable to the problem

        Args:
            variable (Variable): a variable to add

        Raises:
            ValueError: if ``variable`` is not a valid Variable object
        """
        if not isinstance(variable, Variable):
            raise ValueError("Can only add Variable objects")
        self._freeVarIndexMap[variable] = None

    def rmVariable(self, variable):
        """
        Remove a variable from the problem

        Args:
            variable (Variable): the variable to remove. If the variable is not
            part of the problem, no action is taken.
        """
        if variable in self._freeVarIndexMap:
            del self._freeVarIndexMap[variable]
        else:
            logger.error(f"Could not remove variable {variable}")

    def clearVariables(self):
        """
        Remove all variables from the problem
        """
        self._freeVarIndexMap = {}

    def freeVarVec(self, recompute):
        """
        Get the free variable vector

        Args:
            recompute (bool): whether or not to recompute the free variable vector.
                The :func:`freeVarIndexMap` function must be called first (with
                ``recompute = True``).

        Returns:
            numpy.ndarray<float>: the free variable vector
        """
        if recompute:
            self._freeVarVec = np.zeros((self.numFreeVars,))
            for var, ix in self._freeVarIndexMap.items():
                self._freeVarVec[ix : ix + var.numFree] = var.values[~var.values.mask]

        return self._freeVarVec

    def freeVarIndexMap(self, recompute):
        """
        Get the free variable index map

        Args:
            recompute (bool): whether or not to recompute the free variable index
                map, e.g., after variables have been added or removed from the
                problem

        Returns:
            dict: a dictionary mapping the variables included in the problem
            (as :class:`Variable` objects) to the index within the free variable
            vector (as an :class:`int`).
        """
        if recompute:
            # TODO sort variables by type?
            count = 0
            for var in self._freeVarIndexMap:
                self._freeVarIndexMap[var] = count
                count += var.numFree

        return self._freeVarIndexMap

    @property
    def numFreeVars(self):
        """
        Get the number of scalar free variable values in the problem. This is
        distinct from the number of :class:`Variable` objects as each object
        can contain a vector of values and a mask that removes some of the values
        from the free variable vector

        Returns:
            int: number of free variables
        """
        return sum([var.numFree for var in self._freeVarIndexMap])

    # TODO (internal?) function to update variable objects with values from
    #   freeVarVec? Not sure if needed

    # -------------------------------------------
    # Constraints

    def addConstraint(self, constraint):
        """
        Add a constraint to the problem

        Args:
            constraint (AbstractConstraint): a constraint to add
        """
        if not isinstance(constraint, AbstractConstraint):
            raise ValueError("Can only add AbstractConstraint objects")
        self._constraintIndexMap[constraint] = None

    def rmConstraint(self, constraint):
        """
        Remove a constraint from the problem

        Args:
            constraint (AbstractConstraint): the constraint to remove. If the
                constraint is not part of the problem, no action is taken.
        """
        if constraint in self._constraintIndexMap:
            del self._constraintIndexMap[constraint]
        else:
            logger.error(f"Could not remove constraint {constraint}")

    def clearConstraints(self):
        """
        Remove all constraints from the problem
        """
        self._constraintIndexMap = {}

    def constraintVec(self, recompute):
        """
        Get the constraint vector

        Args:
            recompile (bool): whether or not to recompute the constraint vector.
                The :func:`constraintIndexMap` function and :func:`freeVarIndexMap`
                must be called first (with ``recompute = True``).

        Returns:
            numpy.ndarray<float>: the constraint vector
        """
        if recompute:
            self._constraintVec = np.zeros((self.numConstraints,))
            for constraint, ix in self._constraintIndexMap.items():
                self._constraintVec[ix : ix + constraint.size] = constraint.evaluate(
                    self._freeVarIndexMap, self._freeVarVec
                )
        return self._constraintVec

    def constraintIndexMap(self, recompute):
        """
        Get the constraint index map

        Args:
            recompute (bool): whether or not to recompute the constraint index
                map, e.g., after constraints have been added or removed from
                the problem

        Returns:
            dict: a dictionary mapping the constraints included in the problem
                (as objects derived from :class:`AbstractConstraint`) to the index
                within the constraint vector (as an :class:`int`).
        """
        if recompute:
            # TODO sort constraints by type?
            count = 0
            for con in self._constraintIndexMap:
                self._constraintIndexMap[con] = count
                count += con.size

        return self._constraintIndexMap

    @property
    def numConstraints(self):
        """
        Get the number of scalar constraint equations. This is distinct from the
        number of :class:`AbstractConstraint` objects as each object can define
        a vector of equations.

        Returns:
            int: the number of scalar constraint equations
        """
        return sum([con.size for con in self._constraintIndexMap])

    # -------------------------------------------
    # Jacobian

    def jacobian(self, recompute):
        if recompute:
            # Loop through constraints and compute partials with respect to all
            #   of the free variables
            for constraint, cix in self._constraintIndexMap.items():
                # Compute the partials of the constraint with respect to the free
                #   variables
                partials = constraint.partials(self._freeVarIndexMap, self._freeVarVec)

                for partialVar, partialMat in partials.items():
                    # Mask the partials to remove columns associated with variables
                    #   that are not free variables
                    maskedMat = partialVar.maskPartials(partialMat)

                    if maskedMat.size > 0:
                        # TODO insert partialMat into Jacobian
                        pass

        return self._jacobian
