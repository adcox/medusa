"""
Core objects for differential corrections
"""
import logging
import warnings
from abc import ABC, abstractmethod
from copy import copy, deepcopy

import numpy as np
import numpy.ma as ma
import scipy
from rich.columns import Columns
from rich.panel import Panel
from rich.table import Table

from medusa import console, numerics, util
from medusa.dynamics import AbstractDynamicsModel, ModelBlockCopyMixin, VarGroups
from medusa.propagate import Propagator

logger = logging.getLogger(__name__)

__all__ = [
    # base module
    "AbstractConstraint",
    "CorrectionsProblem",
    "ControlPoint",
    "DifferentialCorrector",
    "Segment",
    "Variable",
    "MinimumNormUpdate",
    "LeastSquaresUpdate",
    "L2NormConverged",
    "ShootingProblem",
    # submodules
    "constraints",
]

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
        values (float, [float], numpy.ndarray of float): scalar or array of variable
            values
        mask (bool, [bool], numpy.ndarray of bool): ``True`` flags values as excluded
            from the free variable vector; ``False`` flags values as included
    """

    def __init__(self, values, mask=False, name=""):
        self.values = ma.array(values, mask=mask, ndmin=1)
        self.name = name

    def __repr__(self):
        return "<Variable {!r}, values={!r}>".format(self.name, self.values)

    @property
    def allVals(self):
        return self.values.data

    @property
    def freeVals(self):
        return self.values[~self.values.mask]

    @property
    def mask(self):
        return self.values.mask

    @property
    def numFree(self):
        """
        Get the number of un-masked values, i.e., the number of free variables
        within the vector

        Returns:
            int: the number of un-masked values
        """
        return int(sum(~self.values.mask))

    def unmaskedIndices(self, indices):
        """
        Get the indices of the

        Args:
            indices ([int]): the indices of the Variable value regardless of masking

        Returns:
            [int]: the indices of the requested Variable values within the unmasked
            array.

        Examples:
            >>> Variable([0, 1], [True, False]).unmaskedIndices([1])
            ... [0]
            >>> Variable([0, 1], [True, False]).unmaskedIndices([0])
            ... []
        """
        count = 0
        out = []
        for ix, mask in enumerate(self.values.mask):
            if ix in indices and not mask:
                out.append(count)
            count += not mask

        return out


class AbstractConstraint(ModelBlockCopyMixin, ABC):
    """
    Defines the interface for a constraint object
    """

    def __str__(self):
        """Default to-string implementation"""
        return f"<{type(self).__name__}>"

    @property
    @abstractmethod
    def size(self):
        """
        Get the number of scalar constraint equations

        Returns:
            int: the number of constraint equations
        """
        pass

    def clearCache(self):
        """
        Called by correctors when variables are updated to signal a constraint
        that any caches ought to be cleared and re-computed.

        The default implementation of this method does nothing; derived classes
        should implement it if they cache data.
        """
        pass

    @abstractmethod
    def evaluate(self):
        """
        Evaluate the constraint

        Returns:
            numpy.ndarray of float: the value of the constraint funection; evaluates
            to zero when the constraint is satisfied

        Raises:
            RuntimeError: if the constraint cannot be evaluated
        """
        pass

    @abstractmethod
    def partials(self, freeVarIndexMap):
        """
        Compute the partial derivatives of the constraint vector with respect to
        all variables

        Args:
            freeVarIndexMap (dict): maps the first index (:class:`int`) of each⋅
                :class:`Variable` within ``freeVars`` to the variable object

        Returns:
            dict: a dictionary mapping a :class:`Variable` object to the partial
            derivatives of this constraint with respect to that variable
            (:class:`numpy.ndarray` of :class:`float`). The partial derivatives with respect
            to variables that are not included in the returned dict are assumed
            to be zero.
        """
        # TODO document that the partials should not be masked by free variable
        #    masks; that is accomplished by the jacobian() function
        pass


# ------------------------------------------------------------------------------
# Representing Trajectories
#
#   - ControlPoint
#   - Segment
# ------------------------------------------------------------------------------


class ControlPoint(ModelBlockCopyMixin):
    """
    Defines a propagation start point

    Args:
        model (AbstractDynamicsModel): defines the dynamics model for the propagation
        epoch (float, Variable): the epoch at which the propagation begins. An
            input ``float`` is converted to a :class:`Variable` with name "Epoch"
        state ([float], Variable): the state at which the propagation begins. An
            input list of floats is converted to a :class:`Variable` with name "State"
        autoMask (Optional, bool): whether or not to auto-mask the ``epoch`` variable.
            If True and ``model``
            :func:`~medusa.dynamics.AbstractDynamicsModel.epochIndependent` is True,
            the ``epoch`` variable has its mask set to True.

    Raises:
        TypeError: if the model is not derived from AbstractDynamicsModel
        RuntimeError: if the epoch specifies more than one value
        RuntimeError: if the state specifies more values than the dynamics model
            allows for :attr:`VarGroups.STATE`
    """

    def __init__(self, model, epoch, state, autoMask=True):
        if not isinstance(model, AbstractDynamicsModel):
            raise TypeError("Model must be derived from AbstractDynamicsModel")

        if not isinstance(epoch, Variable):
            # TODO auto-set mask based on model.isTimeDependent()?
            epoch = Variable(epoch, name="Epoch")

        if not isinstance(state, Variable):
            state = Variable(state, name="State")

        if not epoch.values.size == 1:
            raise RuntimeError("Epoch can only have one value")

        sz = model.stateSize(VarGroups.STATE)
        if not state.values.size == sz:
            raise RuntimeError("State must have {sz} values")

        if autoMask:
            epoch.values.mask = model.epochIndependent

        self.model = model
        self.epoch = epoch
        self.state = state

    @property
    def importableVars(self):
        # Define variables attribute for CorrectionsProblem.importVariables
        # Defined as property so that evaluation occurs at runtime, allowing
        #   user to modify/reassign the variables
        return self.state, self.epoch

    @staticmethod
    def fromProp(solution, ix=0, autoMask=True):
        """
        Construct a control point from a propagated arc

        Args:
            solution (scipy.integrate.OptimizeResult): the output from the propagation
            ix (Optional, int): the index of the point within the ``solution``
            autoMask (Optional, bool): whether or not to auto-mask the ``epoch``
                variable. If True and ``solution.model``
                :func:`~medusa.dynamics.AbstractDynamicsModel.epochIndependent`
                is True, the ``epoch`` variable has its mask set to True.

        Returns:
            ControlPoint: a control point with epoch and state retrieved from the
            propagation solution
        """
        if not isinstance(solution, scipy.optimize.OptimizeResult):
            raise TypeError("Expecting OptimizeResult from scipy solve_ivp")

        if ix > len(solution.t):
            raise ValueError(f"ix = {ix} is out of bounds (max = {len(solution.t)})")

        return ControlPoint(solution.model, solution.t[ix], solution.y[:, ix], autoMask)


class Segment:
    """
    Defines a numerical propagation starting from an origin control point for a
    specified time-of-flight.

    Args:
        origin (ControlPoint): the starting point for the propagated trajectory.
            This control point defines the dynamics model, the initial state, and
            the initial epoch.
        tof (float, Variable): the time-of-flight for the propagated trajectory
            in units consistent with the origin's model and state. If the input
            is a float, a :class:`Variable` object with name "Time-of-flight"
            is created.
        terminus (Optional, ControlPoint): a control point located at the terminus
            of the propagated arc. This is useful when defining differential
            corrections problems that seek to constrain the numerical integration.
            It is not required that the terminus state or epoch are numerically
            consistent with the state or epoch values at the end of the arc. The
            ``terminus`` is not used or modified by the Segment.
        prop (Optional, Propagator): the propagator to use. If ``None``, a
            :class:`~medusa.propagate.Propagator` is constructed for the ``origin``
            model with the ``dense`` flag set to False.
        propParams (Optional, list, numpy.ndarray, Variable): parameters to be
            passed to the model's equations of motion. If the input is not a
            :class:`Variable`, a variable is constructed with a name "Params"

    Attributes:
        origin (ControlPoint): the starting point for the propagated trajectory
        terminus (ControlPoint): the terminal point for the propagated trajectory.
            This object is *not* updated by the Segment.
        tof (Variable): the time-of-flight
        prop (Propagtor): the propagator
        propParams (Variable): propagation parameters
        propSol (scipy.optimize.OptimizeResult): the propagation output, if
            :func:`propagate` has been called, otherwise ``None``
    """

    def __init__(self, origin, tof, terminus=None, prop=None, propParams=[]):
        if not isinstance(origin, ControlPoint):
            raise TypeError("origin must be a ControlPoint")

        if terminus is not None and not isinstance(terminus, ControlPoint):
            raise TypeError("terminus must be a ControlPoint or None")

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
        else:
            if not tof.values.size == 1:
                raise RuntimeError("Time-of-flight variable must define only one value")

        if not isinstance(propParams, Variable):
            propParams = Variable(propParams, name="Params")

        self.origin = origin
        self.terminus = terminus
        self.tof = tof
        self.prop = prop
        self.propParams = propParams
        self.propSol = None

    @property
    def importableVars(self):
        # Define variables attribute for CorrectionsProblem.importVariables
        # Defined as property so that evaluation occurs at runtime, allowing
        #   user to modify/reassign the variables
        return self.tof, self.propParams

    def resetProp(self):
        """
        Reset the propagation cache, :attr:`propSol`, to ``None``
        """
        self.propSol = None

    def state(self, ix=-1):
        """
        Lazily propagate the trajectory and retrieve the state

        Args:
            ix (int): the index within the :attr:`propSol` ``y`` array.

        Returns:
            numpy.ndarray: the propagated state (:class:`VarGroups` ``STATE``) on the
            propagated trajectory
        """
        self.propagate(VarGroups.STATE)
        return self.origin.model.extractVars(self.propSol.y[:, ix], VarGroups.STATE)

    def partials_state_wrt_time(self, ix=-1):
        """
        Lazily propagate the trajectory and retrieve the partial derivatives

        Args:
            ix (int): the index within the :attr:`propSol` ``y`` array.

        Returns:
            numpy.ndarray: the partials of the propagated state with respect to
            :attr:`tof`, i.e., the time derivative of the
            :attr:`~medusa.dynamics.VarGroups.STATE` variables
        """
        self.propagate(VarGroups.STATE)
        dy_dt = self.origin.model.diffEqs(
            self.propSol.t[ix],
            self.propSol.y[:, ix],
            (VarGroups.STATE,),
            self.propParams.allVals,
        )
        return self.origin.model.extractVars(dy_dt, VarGroups.STATE)

    def partials_state_wrt_initialState(self, ix=-1):
        """
        Lazily propagate the trajectory and retrieve the partial derivatives

        Args:
            ix (int): the index within the :attr:`propSol` ``y`` array.

        Returns:
            numpy.ndarray: the partials of the propagated state with respect to the
            :attr:`origin` ``state``, i.e., the :attr:`~medusa.dynamics.VarGroups.STM`.
            The partials are returned in matrix form.
        """
        self.propagate([VarGroups.STATE, VarGroups.STM])
        return self.origin.model.extractVars(self.propSol.y[:, ix], VarGroups.STM)

    def partials_state_wrt_epoch(self, ix=-1):
        """
        Lazily propagate the trajectory and retrieve the partial derivatives

        Args:
            ix (int): the index within the :attr:`propSol` ``y`` array.

        Returns:
            numpy.ndarray: the partials of the propagated state with respect to
            the :attr:`origin` ``epoch``, i.e., the
            :attr:`~medusa.dynamics.VarGroups.EPOCH_PARTIALS`.
        """
        self.propagate([VarGroups.STATE, VarGroups.STM, VarGroups.EPOCH_PARTIALS])
        partials = self.origin.model.extractVars(
            self.propSol.y[:, ix], VarGroups.EPOCH_PARTIALS
        )

        # Handle models that don't depend on epoch by setting partials to zero
        if partials.size == 0:
            partials = np.zeros((self.origin.model.stateSize(VarGroups.STATE),))

        return partials

    def partials_state_wrt_params(self, ix=-1):
        """
        Lazily propagate the trajectory and retrieve the partial derivatives

        Args:
            ix (int): the index within the :attr:`propSol` ``y`` array.

        Returns:
            numpy.ndarray: the partials of the propagated state with respect to
            the :attr:`propParams`, i.e., the :attr:`~medusa.dynamics.VarGroups.PARAM_PARTIALS`.
        """
        self.propagate(
            [
                VarGroups.STATE,
                VarGroups.STM,
                VarGroups.EPOCH_PARTIALS,
                VarGroups.PARAM_PARTIALS,
            ]
        )
        partials = self.origin.model.extractVars(
            self.propSol.y[:, ix], VarGroups.PARAM_PARTIALS
        )

        # Handle models that don't depend on propagator params by setting partials
        # to zero
        if partials.size == 0:
            partials = np.zeros((self.origin.model.stateSize(VarGroups.STATE),))

        return partials

    def propagate(self, varGroups, lazy=True):
        """
        Propagate from the :attr:`origin` for the specified :attr:`tof`. Results
        are stored in :attr:`propSol`.

        Args:
            varGroups ([VarGroups]): defines which equations of motion should be included
                in the propagation.
            lazy (Optional, bool): whether or not to lazily propagate the
                trajectory. If True, the :attr:`prop` ``propagate()`` function is
                called if :attr:`propSol` is ``None`` or if the previous propagation
                did not include one or more of the ``varGroups``.

        Returns:
            scipy.optimize.OptimizeResult: the propagation result. This is also
            stored in :attr:`propSol`.
        """
        # Check to see if we can skip the propagation
        if lazy and self.propSol is not None:
            varGroups = np.array(varGroups, ndmin=1)
            if all([v in self.propSol.varGroups for v in varGroups]):
                return self.propSol

        # Propagate from the origin for TOF, set self.propSol
        tspan = [0, self.tof.allVals[0]] + self.origin.epoch.allVals[0]
        self.propSol = self.prop.propagate(
            self.origin.state.allVals,
            tspan,
            params=self.propParams.allVals,
            varGroups=varGroups,
        )

        return self.propSol


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
        # free variables and constraints are stored in a list for add/remove
        self._freeVars = []
        self._constraints = []

        # Other data objects are initialized to None and recomputed on demand
        self._freeVarIndexMap = None
        self._constraintIndexMap = None

        self._freeVarVec = None
        self._constraintVec = None
        self._jacobian = None

    # -------------------------------------------
    # Variables

    def addVariables(self, variable):
        """
        Add one or more variables to the problem

        This clears the caches for :func:`freeVarIndexMap`, :func:`freeVarVec`,
        and :func:`jacobian`.

        Args:
            variable (Variable, [Variable]): one or more variables to add.
                Variables are stored by reference (they are not copied).

        Raises:
            ValueError: if any of the inputs are not :class:`Variable: objects
        """
        for var in util.toList(variable):
            if not isinstance(var, Variable):
                raise ValueError("Can only add Variable objects")

            if var.numFree == 0:
                logger.debug(f"Cannot add {var}; it has no free values")
                continue

            if var in self._freeVars:
                logger.debug(f"Skipping add {var}; it has already been added")
                continue

            self._freeVars.append(var)
            self._freeVarIndexMap = None
            self._freeVarVec = None
            self._jacobian = None

    def importVariables(self, obj):
        """
        Imports variables from an object. The object must define a ``importableVars``
        attribute that contains any variables that can be imported by the problem.
        Variables are added via :func:`addVariable`, clearing the caches if
        applicable.

        Args:
            obj: an object
        """
        self.addVariables(util.toList(getattr(obj, "importableVars", [])))

    def rmVariables(self, variable):
        """
        Remove one or more variables from the problem

        This clears the caches for :func:`freeVarIndexMap`, :func:`freeVarVec`,
        and :func:`jacobian`.

        Args:
            variable (Variable, [Variable]): the variable(s) to remove. If the
                variable is not part of the problem, no action is taken.
        """
        for var in util.toList(variable):
            try:
                self._freeVars.remove(var)
                self._freeVarIndexMap = None
                self._freeVarVec = None
                self._jacobian = None
            except ValueError:
                logger.debug(f"Could not remove variable {var}")

    def clearVariables(self):
        """
        Remove all variables from the problem

        This clears the caches for :func:`freeVarIndexMap`, :func:`freeVarVec`,
        and :func:`jacobian`.
        """
        self._freeVars = []
        self._freeVarIndexMap = None
        self._freeVarVec = None
        self._jacobian = None

    def freeVarVec(self):
        """
        Get the free variable vector

        Returns:
            numpy.ndarray<float>: the free variable vector. This result is cached
            until the free variables are updated.
        """
        if self._freeVarVec is None:
            self._freeVarVec = np.zeros((self.numFreeVars,))
            for var, ix in self.freeVarIndexMap().items():
                self._freeVarVec[ix : ix + var.numFree] = var.values[~var.values.mask]

        return self._freeVarVec

    def freeVarIndexMap(self):
        """
        Get the free variable index map

        Returns:
            dict: a dictionary mapping the variables included in the problem
            (as :class:`Variable` objects) to the index within the free variable
            vector (as an :class:`int`). This result is cached until the free
            variables are modified.
        """
        if self._freeVarIndexMap is None:
            # TODO sort variables by type?
            self._freeVarIndexMap = {}
            count = 0
            for var in self._freeVars:
                self._freeVarIndexMap[var] = count
                count += var.numFree

        return self._freeVarIndexMap

    def freeVarIndexMap_reversed(self):
        varMap = {}
        for var, ix0 in self.freeVarIndexMap().items():
            for k in range(var.numFree):
                assert (
                    not ix0 + k in varMap
                ), "duplicate entry in reversed freeVarIndexMap"
                varMap[ix0 + k] = var
        return varMap

    def updateFreeVars(self, newVec):
        """
        Update the free variable vector and corresponding Variable objects

        This clears the caches for :func:`freeVarVec`, :func:`constraintVec`,
        and :func:`jacobian`. The :func:`AbstractConstraint.clearCache` method is
        also called on all constraints in the problem.

        Args:
            newVec (numpy.ndarray): an updated free variable vector. It must
                be the same size and shape as the existing free variable vector.

        Raises:
            ValueError: if ``newVec`` doesn't have the same shape as the
                existing free variable vector
        """
        if not newVec.shape == self.freeVarVec().shape:
            raise ValueError(
                f"Shape of newVec {newVec.shape} doesn't match existing "
                f"free variable vector {self.freeVarVec().shape}"
            )

        for var, ix0 in self.freeVarIndexMap().items():
            var.values[~var.mask] = newVec[ix0 : ix0 + var.numFree]

        self._freeVarVec = None
        self._constraintVec = None
        self._jacobian = None
        for constraint in self._constraints:
            constraint.clearCache()

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
        return sum([var.numFree for var in self._freeVars])

    # -------------------------------------------
    # Constraints

    def addConstraints(self, constraint):
        """
        Add one or more constraints to the problem. If the constraint defines variables,
        they are also added to the problem via :func:`importVariables`.

        This clears the caches for :func:`constraintIndexMap`,
        :func:`constraintVec`, and :func:`jacobian`

        Args:
            constraint (AbstractConstraint, [AbstractConstraint]): one or more
                constraints to add. Constraints are stored by reference (they are
                not copied).
        """
        for con in util.toList(constraint):
            # TODO allow input to be an iterable
            if not isinstance(con, AbstractConstraint):
                raise ValueError("Can only add AbstractConstraint objects")

            if con.size == 0:
                logger.debug(f"Cannot add {con}; its size is zero")
                continue

            if con in self._constraints:
                logger.debug(f"Constraint {con} has laredy been added")
                continue

            self._constraints.append(con)
            self.importVariables(con)

            self._constraintIndexMap = None
            self._constraintVec = None
            self._jacobian = None

    def rmConstraints(self, constraint):
        """
        Remove one or more constraints from the problem. If the constraint defines
        variables, they are also removed from the problem.

        This clears the caches for :func:`constraintIndexMap`,
        :func:`constraintVec`, and :func:`jacobian`

        Args:
            constraint (AbstractConstraint, [AbstractConstraint]): the constraint(s)
                to remove. If the constraint is not part of the problem, no action
                is taken.
        """
        for con in util.toList(constraint):
            try:
                self._constraints.remove(con)
                self.rmVariables(util.toList(getattr(con, "importableVars", [])))
                self._constraintIndexMap = None
                self._constraintVec = None
                self._jacobian = None
            except ValueError:
                logger.debug(f"Could not remove constraint {con}")

    def clearConstraints(self):
        """
        Remove all constraints from the problem.

        This clears the caches for :func:`constraintIndexMap`,
        :func:`constraintVec`, and :func:`jacobian`
        """
        self._constraints = []
        self._constraintIndexMap = None
        self._constraintVec = None
        self._jacobian = None

    def constraintVec(self):
        """
        Get the constraint vector

        Args:
            recompile (bool): whether or not to recompute the constraint vector.
                The :func:`constraintIndexMap` function and :func:`freeVarIndexMap`
                must be called first (with ``recompute = True``).

        Returns:
            numpy.ndarray<float>: the constraint vector. This result is cached
            until the free variables or constraints are updated.
        """
        if self._constraintVec is None:
            self._constraintVec = np.zeros((self.numConstraints,))
            for constraint, ix in self.constraintIndexMap().items():
                self._constraintVec[ix : ix + constraint.size] = constraint.evaluate()
        return self._constraintVec

    def constraintIndexMap(self):
        """
        Get the constraint index map

        Args:
            recompute (bool): whether or not to recompute the constraint index
                map, e.g., after constraints have been added or removed from
                the problem

        Returns:
            dict: a dictionary mapping the constraints included in the problem
                (as objects derived from :class:`AbstractConstraint`) to the index
                within the constraint vector (as an :class:`int`). This result is
                cached until the constraints are updated.
        """
        if self._constraintIndexMap is None:
            # TODO sort constraints by type?
            self._constraintIndexMap = {}
            count = 0
            for con in self._constraints:
                self._constraintIndexMap[con] = count
                count += con.size

        return self._constraintIndexMap

    def constraintIndexMap_reversed(self):
        conMap = {}
        for con, ix0 in self.constraintIndexMap().items():
            for k in range(con.size):
                assert (
                    not ix0 + k in conMap
                ), "duplicate entry in reversed constraintIndexMap"
                conMap[ix0 + k] = con
        return conMap

    @property
    def numConstraints(self):
        """
        Get the number of scalar constraint equations. This is distinct from the
        number of :class:`AbstractConstraint` objects as each object can define
        a vector of equations.

        Returns:
            int: the number of scalar constraint equations
        """
        return sum([con.size for con in self._constraints])

    # -------------------------------------------
    # Jacobian

    def jacobian(self, numeric=False, stepSize=1e-4):
        """
        Get the Jacobian matrix, i.e., the partial derivative of the constraint
        vector with respect to the free variable vector. The rows of the matrix
        correspond to the scalar constraints and the columns of the matrix
        correspond to the scalar free variables.

        Args:
            numeric (bool, Optional): if False, the Jacobian is computed analytically.
                Otherwise, the :func:`medusa.numerics.derivative_multivar` method is
                used to compute the Jacobian numerically.
            stepSize (float, Optional): if ``numeric`` is True, this is the step
                size for the ``derivative_multivar`` function.

        Returns:
            numpy.ndarray: the Jacobian matrix. This result is cached until either
            the free variables or constraints are updated.
        """
        if numeric:
            prob = deepcopy(self)

            # A simple function to evaluate the constraint vector
            def evalCons(freeVars):
                prob.updateFreeVars(freeVars)
                return copy(prob.constraintVec())

            return numerics.derivative_multivar(
                evalCons, copy(prob.freeVarVec()), stepSize
            )
            # TODO cache the numerical jacobian separately??

        if self._jacobian is None:
            self._jacobian = np.zeros((self.numConstraints, self.numFreeVars))
            # Loop through constraints and compute partials with respect to all
            #   of the free variables
            for constraint, cix in self.constraintIndexMap().items():
                # Compute the partials of the constraint with respect to the free
                #   variables
                partials = constraint.partials(self.freeVarIndexMap())

                for partialVar, partialMat in partials.items():
                    if not partialVar in self.freeVarIndexMap():
                        continue
                    # Mask the partials to remove columns associated with variables
                    #   that are not free variables
                    if any(~partialVar.values.mask):
                        cols = self.freeVarIndexMap()[partialVar] + np.arange(
                            partialVar.numFree
                        )
                        for rix, partials in zip(
                            cix + np.arange(constraint.size), partialMat
                        ):
                            if len(partialMat.shape) > 1:
                                self._jacobian[rix, cols] = partials[
                                    ~partialVar.values.mask
                                ]
                            elif not partialVar.values.mask[0]:
                                self._jacobian[rix, cols] = partials

        return self._jacobian

    def checkJacobian(self, stepSize=1e-4, tol=1e-6):
        """
        Compute a numerical Jacobian and compare against the analytically-computed
        one.

        The "Richardson's deferred approach to the limit" algorithm, implemented in
        :func:`~medusa.numerics.derivative`, is employed to compute the numerical
        version of the Jacobian matrix. Each element of the free variable vector
        is perturbed, yielding a similarly perturbed constraint vector. By
        differencing these constraint vectors, a derivative with respect to the
        perturbed free variable is obtained, i.e., a column of the Jacobian matrix.

        The numerical Jacobian matrix is then differenced from the analytical
        Jacobian (from :func:`jacobian`). For non-zero Jacobian entries with absolute
        values larger than the step size, the relative difference is computed as
        ``(numeric - analytical)/numeric``. For other Jacobian entries, the
        absolute difference, ``(numeric - analytical)`` is stored. The two are
        deemed equal if each difference is less than ``tol``. Messages about
        inequality are logged to the "medusa.corrections" logger at the ERROR level.

        This is not a foolproof method (numerics can be tricky) but is often a
        helpful tool in debugging partial derivative derivations.

        Args:
            stepSize (Optional, float): the initial free variable step size; this
            value should be sufficiently large such that the constraint vector
            changes *substantially* as a result.
            tol (Optional, float): tolerance for equality between the numeric and
                analytical Jacobian matrices

        Returns:
            bool: True if the numeric and analytical Jacobian matrices are equal
        """
        analytic = self.jacobian()
        prob = deepcopy(self)

        # A simple function to evaluate the constraint vector
        def evalCons(freeVars):
            prob.updateFreeVars(freeVars)
            return copy(prob.constraintVec())

        numeric = numerics.derivative_multivar(
            evalCons, copy(prob.freeVarVec()), stepSize
        )

        # Map indices to variable and constraint objects; used for better printouts
        varMap = self.freeVarIndexMap_reversed()
        conMap = self.constraintIndexMap_reversed()

        # Compute absolute and relative differences and determine equality
        absDiff = numeric - analytic
        relDiff = absDiff.copy()
        equal = True
        for r, row in enumerate(relDiff):
            for c in range(len(row)):
                # If numeric partial is nonzero, compute relative dif
                if abs(numeric[r, c]) > 1e-12:
                    relDiff[r, c] = absDiff[r, c] / numeric[r, c]

                if abs(relDiff[r, c]) > tol:
                    equal = False

                    con, var = conMap[r], varMap[c]
                    conDetail = "Constraint (sub-index {}) = {}".format(
                        r - self.constraintIndexMap()[con], con
                    )
                    varDetail = "Variable (sub-index {}) = {}".format(
                        c - self.freeVarIndexMap()[var], var
                    )
                    logger.error(
                        f"Jacobian error at ({r}, {c}): "
                        f"Expected = {numeric[r,c]}, Actual = {analytic[r,c]} "
                        f"(Rel err = {relDiff[r,c]:e}\n  {conDetail}\n  {varDetail}"
                    )

        return equal

    # -------------------------------------------
    # Printing
    def printFreeVars(self):
        """
        Print free variables to the screen
        """
        # TODO include flag to show values?
        columns = Columns(title="Free Variables")
        for var, ix0 in self.freeVarIndexMap().items():
            name = var.name if var.name else var.__class__.__name__
            if var.numFree <= 1:
                indices = f"Index: {ix0}"
            else:
                indices = f"Indices: {ix0} - {ix0+var.numFree-1}"

            columns.add_renderable(Panel(indices, title=f"[b]{name}[/b]"))
        console.print(columns)

    def printConstraints(self):
        """
        Print constraints to the screen
        """
        # TODO include flag to show values?
        columns = Columns(title="Constraints")
        for con, ix0 in self.constraintIndexMap().items():
            name = con.__class__.__name__.replace("Constraint", "")
            if con.size <= 1:
                indices = f"Index: {ix0}"
            else:
                indices = f"Indices: {ix0} - {ix0+con.size-1}"

            columns.add_renderable(Panel(indices, title=f"[b]{name}[/b]"))
        console.print(columns)

    def printJacobian(self, numeric=False):
        """
        Print the Jacobian matrix to the screen
        """
        jacobian = self.jacobian(numeric=numeric)
        varMap = self.freeVarIndexMap_reversed()
        conMap = self.constraintIndexMap_reversed()

        columns = Columns(title="Jacobian", padding=(0, 0))
        for var, ix0 in self.freeVarIndexMap().items():
            name = var.name if var.name else var.__class__.__name__
            header = [f"{ix:02d}" for ix in range(ix0, ix0 + var.numFree)]
            if ix0 == 0:
                header.insert(0, "")
            table = Table(
                *header,
                collapse_padding=True,
                show_edge=False,
                pad_edge=False,
                expand=True,
            )
            lastConIx = 0

            for ix, row in enumerate(jacobian[:, ix0 : ix0 + var.numFree]):
                rowText = [f"[bold]{ix:02d}[/bold]"] if ix0 == 0 else []
                for val in row:
                    if val is None or val == 0.0:
                        rowText.append("[gray50]0[/gray50]")
                    elif np.isnan(val):
                        rowText.append("[red]NaN[/red]")
                    elif int(val) == val:
                        rowText.append(f"[green]{int(val):d}[/green]")
                    elif abs(np.log10(abs(val))) < 3:
                        rowText.append(f"[blue]{val:.2f}[/blue]")
                    else:
                        rowText.append(f"[blue]{val:.2e}[/blue]")

                if ix < jacobian.shape[0] - 1:
                    nextConIx = self._constraints.index(conMap[ix + 1])
                    end_section = nextConIx > lastConIx
                    lastConIx = nextConIx

                table.add_row(*rowText, end_section=end_section)

            columns.add_renderable(Panel(table, title=f"[b]{name}[/b]", expand=False))

        console.print(columns)


class ShootingProblem(CorrectionsProblem):
    """
    A specialized version of the :class:`CorrectionsProblem` that accepts
    :class:`Segment` objects as inputs; variables are automatically extracted
    from the segments and their control points. Additionally, a directed
    graph is constructed to ensure the collection of segments make physical sense.
    """

    # TODO need public access to segments

    def __init__(self):
        super().__init__()

        self._segments, self._points = [], []
        self._adjMat = None  # adjacency matrix

    def addSegments(self, segment):
        """
        Add one or more segments to the problem. Segments are not added if they
        already exist in the problem.

        Args:
            segment (Segment, [Segment]): a single or multiple segments to add
                to the problem

        Raises:
            ValueError: if one of the inputs is not a :class:`Segment`
        """
        for seg in util.toList(segment):
            if not isinstance(seg, Segment):
                raise ValueError("Can only add segment objects")

            if not seg in self._segments:
                self._segments.append(seg)
                self._adjMat = None

    def rmSegments(self, segment):
        """
        Remove one or more segments from the problem. Segments that are not
        in the problem are ignored.
        """
        for seg in util.toList(segment):
            try:
                self._segments.remove(seg)
                self._adjMat = None
            except ValueError:
                logger.debug(f"Can not remove {seg} from segments")

    def build(self):
        """
        Call this method after all segments, variables, and constraints have
        been added to the problem (i.e., right before calling
        :func:`DifferentialCorrector.solve`).

        The control points from each segment are extracted and their free variables
        added to the problem. Similarly, the segment free variables are added to
        the problem. The :func:`checkValidGraph` function is called afteward to
        validate the configuration.

        Raises:
            RuntimeError: if the graph validation fails
        """
        # Clear caches
        self._freeVarIndexMap = None
        self._freeVarVec = None
        self._jacobian = None
        self._points = []
        self._adjMat = None

        # Loop through segments and add all variables
        for seg in self._segments:
            # Add control points to list and import their variables
            if not seg.origin in self._points:
                self._points.append(seg.origin)
                self.importVariables(seg.origin)

            if seg.terminus and not seg.terminus in self._points:
                self._points.append(seg.terminus)
                self.importVariables(seg.terminus)

            self.importVariables(seg)

        # Create directed graph
        errors = self.checkValidGraph()
        if len(errors) > 0:
            for err in errors:
                logger.error(err)
            raise RuntimeError("Invalid directed graph")

        # TODO set flag for built = True?

    def adjacencyMatrix(self):
        """
        Create an adjacency matrix for the directed graph of control points and
        segments.

        Define the adjacency matrix as ``A[row, col]``; each row and column corresponds
        to a control point in the internal storage list. The value of ``A[r, c]``
        is the index of a segment within the segment storage list with an origin
        at control point ``r`` and a terminus at control point ``c`` where
        ``r`` and ``c`` are the indices of those points within the storage list.
        A ``None`` value in the adjacency matrix indicates that there is not a
        segment linking the two control points.

        Returns:
            numpy.ndarray: the adjacency matrix. This value is cached and reset
            whenever segments are added or removed.

        Raises:
            ValueError: if an origin or terminus control point cannot be located
                within the storage list
        """
        if self._adjMat is None:
            # fill with None
            N = len(self._points)
            if N == 0:
                raise RuntimeError(
                    "No control points have been added; did you call build()?"
                )

            self._adjMat = np.full((N, N), None)

            # Put segment indices in the spots that link control points
            for segIx, seg in enumerate(self._segments):
                try:
                    ixO = self._points.index(seg.origin)
                except ValueError:
                    raise RuntimeError(
                        f"{seg} origin ({seg.origin}) has not been added to problem"
                    )

                try:
                    ixT = self._points.index(seg.terminus)
                except ValueError:
                    raise RuntimeError(
                        f"{seg} terminus ({seg.terminus}) has not been added to problem"
                    )

                self._adjMat[ixO][ixT] = segIx

        return self._adjMat

    def checkValidGraph(self):
        """
        Check that the directed graph (adjacency matrix) is valid. Four rules apply:

        1. Each origin point can be linked to a maximum of 2 segments
        2. If 2 segments are linked to the same origin, they must have opposite TOF signs
        3. Each terminal point can be linked to a maximum of 1 segment
        4. Each control point must be linked to a segment (no floating points)
        """
        adjMat = self.adjacencyMatrix()
        errors = []
        pointIsLinked = np.full(len(self._points), False)

        # Check origin control points
        for r, row in enumerate(adjMat):
            segCount = 0
            tofSignSum = 0
            for c in range(len(row)):
                if adjMat[r, c] is not None:
                    segCount += 1
                    tof = self._segments[adjMat[r, c]].tof
                    tofSignSum += int(np.sign(tof.allVals)[0])
                    pointIsLinked[r] = True

                    # A segment cannot link a point to itself!
                    if r == c:
                        errors.append(
                            f"Segment {adjMat[r,c]} links point {r} to itself"
                        )

            # An origin can have, at most, 2 segments
            if segCount > 2:
                errors.append(
                    f"Origin control point {r} has {segCount} linked segments but must have <= 2"
                )
            elif segCount == 2 and not tofSignSum == 0:
                # TOF signs must be different
                errors.append(
                    f"Origin control point {r} is linked to two segments with sign(tof) = {tofSignSum/2}"
                )

        # Check terminal control points
        for c in range(adjMat.shape[1]):
            segCount = 0
            for r in range(adjMat.shape[0]):
                if adjMat[r, c] is not None:
                    segCount += 1
                    pointIsLinked[c] = True

            # Only 1 segment can be linked to terminal node; 2 segments that end
            #   at the same node is not allowed
            if segCount > 1:
                errors.append(
                    f"Terminal control point {c} has {segCount} linked segments but must have <= 1"
                )

        # Check for floating (unlinked) points
        for ix, val in enumerate(pointIsLinked):
            if not val:
                errors.append(f"Control point {r} is not linked to any segments")

        return errors


class DifferentialCorrector:
    """
    Apply a differential corrections method to solve a corrections problem

    Attributes:
        maxIterations (int): the maximum number of iterations to attempt
        convergenceCheck (object): an object containing a "isConverged" method
            that accepts a :class:`CorrectionsProblem` as an input and returns
            a :class:`bool`.
        updateGenerator (object): an object containing a "update" method that
            accepts a :class:`CorrectionsProblem` as an input and returns a
            :class:`~numpy.ndarray` representing the change in the free
            variable vector. See :class:`MinimumNormUpdate` for an example.
    """

    def __init__(self):
        self.maxIterations = 20
        self.convergenceCheck = L2NormConvergence(1e-8)
        self.updateGenerator = MinimumNormUpdate()

    def _validateArgs(self):
        """
        Check the types and values of class attributes so that useful errors
        can be thrown before those attributes are evaluated or used.
        """
        # maxIterations
        if not isinstance(self.maxIterations, int):
            raise TypeError("maxIterations must be an integer")
        if not self.maxIterations > 0:
            raise ValueError("maxIterations must be positive")

        # convergence check
        if self.convergenceCheck is None:
            raise TypeError(
                "convergenceCheck is None; please assign a convergence check"
            )
        if not hasattr(self.convergenceCheck, "isConverged"):
            raise AttributeError("convergenceCheck needs a method named 'isConverged'")
        if not callable(self.convergenceCheck.isConverged):
            raise TypeError("convergenceCheck.isConverged must be callable")

        # Update generator
        if self.updateGenerator is None:
            raise TypeError(
                "updateGenerator is None; please assign an update generator"
            )
        if not hasattr(self.updateGenerator, "update"):
            raise AttributeError("updateGenerator needs a method named 'update'")
        if not callable(self.updateGenerator.update):
            raise TypeError("updateGenerator.update must be callable")

    def solve(self, problem):
        """
        Solve a corrections problem by iterating the :attr:`updateGenerator`
        until the :attr:`convergenceCheck` returns True or the :attr:`maxIterations`
        are reached.

        Args:
            problem (CorrectionsProblem): the corrections problem to be solved

        Returns:
            tuple: a Tuple with two elements. The first is a :class:`CorrectionsProblem`
            that stores the solution after the final iteration of the solver. The
            second is a :class:`dict` with the following keywords:

            - ``status`` (:class:`str`): the status of the solver after the final
              iteration. Can be "empty" if the problem contained no free variables
              or constraints, "converged" if the convergence check was satisfied,
              or "max-iterations" if the maximum number of iterations were completed
              before convergence.
            - ``iterations`` (:class:`list`): a list of `dict`; each dict represents
              an iteration of the solver and includes a copy of the free variable
              vector and the constraint vector.
        """
        self._validateArgs()

        solution = deepcopy(problem)
        # TODO clear all caches in solution

        log = {"status": "", "iterations": []}

        if solution.numFreeVars == 0 or solution.numConstraints == 0:
            log["status"] = "empty"
            return solution

        with warnings.catch_warnings(record=True) as caught:
            warnings.filterwarnings("error", module="scipy.linalg")
            itCount = 0
            logger.info("Beginning differential corrections")
            while True:
                if itCount > 0:
                    freeVarStep = self.updateGenerator.update(solution)
                    newVec = solution.freeVarVec() + freeVarStep
                    solution.updateFreeVars(newVec)

                log["iterations"].append(
                    {
                        "free-vars": copy(solution.freeVarVec()),
                        "constraints": copy(solution.constraintVec()),
                    }
                )

                # TODO allow user to customize printout?
                err = np.linalg.norm(solution.constraintVec())
                logger.info(f"Iteration {itCount:03d}: ||F|| = {err:.4e}")

                while len(caught) > 0:
                    record = logger.makeRecord(
                        "warning",
                        logging.WARNING,
                        caught[0].filename,
                        caught[0].lineno,
                        caught[0].message,
                        {},
                        None,
                    )
                    caught.pop(0)
                    logger.handle(record)

                itCount += 1

                if self.convergenceCheck.isConverged(solution):
                    log["status"] = "converged"
                    break
                elif itCount >= self.maxIterations:
                    log["status"] = "max-iterations"
                    break

        return solution, log


class MinimumNormUpdate:
    """
    Computes the minimum-norm update to the problem J @ dX + FX = 0

    TODO link to math spec
    """

    def update(self, problem):
        """
        Compute the minimum-norm update

        Args:
            problem (CorrectionsProblem): the corrections problem to be solved

        Returns:
            numpy.ndarray: the free variable vector step, `dX`
        """
        nFree = problem.numFreeVars
        # using -FX for all equations, so premultiply
        FX = -1 * problem.constraintVec()

        if len(FX) > nFree:
            raise RuntimeError(
                "Minimum Norm Update requires fewer or equal number of "
                "constraints as free variables"
            )

        jacobian = problem.jacobian()
        if len(FX) == nFree:
            # Jacobian is square; it must be full rank!
            # Solve the linear system J @ dX = -FX for dX
            return scipy.linalg.solve(jacobian, FX)
        elif len(FX) < nFree:
            # Compute Gramm matrix; J must have linearly independent rows;
            #    rank(J) == nRows
            JT = jacobian.T

            # Solve (J @ J.T) @ W = -FX for W
            # Minimum norm step is dX = JT @ W
            W = scipy.linalg.solve(jacobian @ JT, FX)
            return JT @ W


class LeastSquaresUpdate:
    """
    Computes the least squares update to the problem J @ dX + FX = 0

    TODO link to math spec
    """

    def update(self, problem):
        """
        Compute the least-squares update

        Args:
            problem (CorrectionsProblem): the corrections problem to be solved

        Returns:
            numpy.ndarray: the free variable vector step, `dX`
        """
        # Using -FX for all equations, so pre-multiply
        FX = -1 * problem.constraintVec()

        if len(FX) <= problem.numFreeVars:
            raise RuntimeError(
                "Least Squares UPdate requires more constraints than free variables"
            )

        # System is over-constrained; J must have linearly independent columns;
        #   rank(J) == nCols
        J = problem.jacobian()
        JT = J.T
        G = JT @ J

        # Solve system (J' @ J) @ dX = -J' @ FX for dX
        return scipy.linalg.solve(G, JT @ FX)


class L2NormConvergence:
    """
    Define convergence via the L2 norm of the constraint vector

    Args:
        tol (float): the maximum L2 norm for convergence
    """

    def __init__(self, tol):
        self.tol = tol

    def isConverged(self, problem):
        """
        Args:
            problem (CorrectionsProblem): the corrections problem to be evaluated

        Returns:
            bool: True if the L2 norm of the :func:`~CorrectionsProblem.constraintVec`
            is less than or equal to ``tol``
        """
        return np.linalg.norm(problem.constraintVec()) <= self.tol
