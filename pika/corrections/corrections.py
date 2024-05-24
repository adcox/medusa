"""
Core Corrections Class
"""
import logging
from abc import ABC, abstractmethod
from copy import copy, deepcopy

import numpy as np
import numpy.ma as ma
import scipy

from pika.dynamics import AbstractDynamicsModel, EOMVars, ModelBlockCopyMixin
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

    @property
    @abstractmethod
    def size(self):
        """
        Get the number of scalar constraint equations

        Returns:
            int: the number of constraint equations
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


class ControlPoint(ModelBlockCopyMixin):
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
        if not isinstance(solution, scipy.optimize.OptimizeResult):
            raise TypeError("Expecting OptimizeResult from scipy solve_ivp")

        if ix > len(solution.t):
            raise ValueError(f"ix = {ix} is out of bounds (max = {len(solution.t)})")

        return ControlPoint(solution.model, solution.t[ix], solution.y[:, ix])


class Segment:
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

        if not isinstance(propParams, Variable):
            propParams = Variable(propParams, name="Params")

        self.origin = origin
        self.terminus = terminus
        self.tof = tof
        self.prop = prop
        self.propParams = propParams
        self.propSol = None

    def finalState(self):
        self.propagate(EOMVars.STATE)
        return self.origin.model.extractVars(self.propSol.y[:, -1], EOMVars.STATE)

    def partials_finalState_wrt_time(self):
        self.propagate(EOMVars.STATE)
        dy_dt = self.origin.model.evalEOMs(
            self.propSol.t[-1],
            self.propSol.y[:, -1],
            [EOMVars.STATE],
            self.propParams.values,
        )
        return self.origin.model.extractVars(dy_dt, EOMVars.STATE)

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
        # free variables and constraints are stored in a list for add/remove
        self._freeVars = []
        self._constraints = []

        # Other data objects are initialized to None and recomputed on demand
        self._freeVarIndexMap = None  # {}
        self._constraintIndexMap = None

        self._freeVarVec = None  # np.empty((0,))
        self._constraintVec = None
        self._jacobian = None

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

        if variable.numFree == 0:
            logger.error(f"Cannot add {variable}; it has no free values")
            return

        if variable in self._freeVars:
            raise RuntimeError("Variable has already been added")

        self._freeVars.append(variable)
        self._freeVarIndexMap = None
        self._freeVarVec = None
        self._jacobian = None

    def rmVariable(self, variable):
        """
        Remove a variable from the problem

        Args:
            variable (Variable): the variable to remove. If the variable is not
            part of the problem, no action is taken.
        """
        try:
            self._freeVars.remove(variable)
            self._freeVarIndexMap = None
            self._freeVarVec = None
            self._jacobian = None
        except ValueError:
            logger.error(f"Could not remove variable {variable}")

    def clearVariables(self):
        """
        Remove all variables from the problem
        """
        self._freeVars = []
        self._freeVarIndexMap = None
        self._freeVarVec = None
        self._jacobian = None

    def freeVarVec(self):
        """
        Get the free variable vector

        Returns:
            numpy.ndarray<float>: the free variable vector
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
            vector (as an :class:`int`).
        """
        if self._freeVarIndexMap is None:
            # TODO sort variables by type?
            self._freeVarIndexMap = {}
            count = 0
            for var in self._freeVars:
                self._freeVarIndexMap[var] = count
                count += var.numFree

        return self._freeVarIndexMap

    def updateFreeVars(self, newVec):
        """
        Update the free variable vector and corresponding Variable objects

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

        if constraint.size == 0:
            logger.error(f"Cannot add {constraint}; its size is zero")
            return

        if constraint in self._constraints:
            raise RuntimeError("Constraint has laredy been added")

        self._constraints.append(constraint)
        self._constraintIndexMap = None
        self._constraintVec = None
        self._jacobian = None

    def rmConstraint(self, constraint):
        """
        Remove a constraint from the problem

        Args:
            constraint (AbstractConstraint): the constraint to remove. If the
                constraint is not part of the problem, no action is taken.
        """
        try:
            self._constraints.remove(constraint)
            self._constraintIndexMap = None
            self._constraintVec = None
            self._jacobian = None
        except ValueError:
            logger.error(f"Could not remove constraint {constraint}")

    def clearConstraints(self):
        """
        Remove all constraints from the problem
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
            numpy.ndarray<float>: the constraint vector
        """
        if self._constraintVec is None:
            self._constraintVec = np.zeros((self.numConstraints,))
            for constraint, ix in self.constraintIndexMap().items():
                self._constraintVec[ix : ix + constraint.size] = constraint.evaluate(
                    self.freeVarIndexMap(), self.freeVarVec()
                )
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
                within the constraint vector (as an :class:`int`).
        """
        if self._constraintIndexMap is None:
            # TODO sort constraints by type?
            self._constraintIndexMap = {}
            count = 0
            for con in self._constraints:
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
        return sum([con.size for con in self._constraints])

    # -------------------------------------------
    # Jacobian

    def jacobian(self):
        if self._jacobian is None:
            self._jacobian = np.zeros((self.numConstraints, self.numFreeVars))
            # Loop through constraints and compute partials with respect to all
            #   of the free variables
            for constraint, cix in self.constraintIndexMap().items():
                # Compute the partials of the constraint with respect to the free
                #   variables
                partials = constraint.partials(
                    self.freeVarIndexMap(), self.freeVarVec()
                )

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
                            self._jacobian[rix, cols] = partials

        return self._jacobian


class DifferentialCorrector:
    def __init__(self):
        self.maxIterations = 20

        self.convergenceCheck = None
        self.updateGenerator = None
        self.solution = None

    def solve(self, problem):
        self.solution = deepcopy(problem)

        if self.solution.numFreeVars == 0 or self.solution.numConstraints == 0:
            return self.solution

        itCount = 0
        while True:
            if itCount > 0:
                freeVarStep = self.updateGenerator(self.solution)
                newVec = self.solution.freeVarVec() + freeVarStep
                self.solution.updateFreeVars(newVec)

            print(self.solution.freeVarVec())
            print(self.solution.constraintVec())

            err = np.linalg.norm(self.solution.constraintVec())
            logger.info(f"Iteration {itCount:03d}: ||F|| = {err:.4e}")
            itCount += 1

            if self.convergenceCheck(self.solution) or itCount >= self.maxIterations:
                break

        if not self.convergenceCheck(self.solution):
            return None

        return self.solution


def minimumNormUpdate(problem):
    nFree = problem.numFreeVars
    FX = -1 * problem.constraintVec()  # using -FX for all equations, so premultiply

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
        # Compute Gramm matrix; J must have linearly independent rows; rank(J) == nRows
        JT = jacobian.T
        G = jacobian @ JT
        W = scipy.linalg.solve(G, FX)  # Solve G @ W = -FX for W
        return JT @ W  # Minimum norm step is dX = JT @ W


def leastSquaresUpdate(problem):
    FX = -1 * problem.constraintVec()  # using -FX for all equations, so premultiply

    if len(FX) <= problem.numFreeVars:
        raise RuntimeError(
            "Least Squares UPdate requires more constraints than free variables"
        )

    # System is over-constrained; J must have linearly independent columns;
    #   rank(J) == nCols
    JT = jacobian.T
    G = JT @ J

    # Solve system (J' @ J) @ dX = -J' @ FX for dX
    return scipy.linalg.solve(G, JT @ FX)


def constraintVecL2Norm(problem):
    return np.linalg.norm(problem.constraintVec()) < 1e-4
