"""
Dynamics Classes and Interfaces
"""
import logging
from abc import ABC, abstractmethod
from copy import copy, deepcopy
from enum import IntEnum

import numba
import numpy as np

from pika.data import Body

logger = logging.getLogger(__name__)


# numba JIT compilation only supports Enum and IntEnum
class VarGroups(IntEnum):
    """
    Specify the variable groups included in a model variable array. The integer
    values of the groups correspond to their location in a variable array. I.e.,
    the ``STATE`` variables are always first, followed by the ``STM``,
    ``EPOCH_PARTIALS``, and ``PARAM_PARTIALS``. All matrix objects are stored in
    row-major order
    """

    STATE = 0
    """
    State variables; usually includes the position and velocity coordinates
    """

    STM = 1
    """ 
    State Transition Matrix; An N**2 matrix of the time-evolving partial derivatives
    of the propagated state w.r.t. the initial state, where N is the size of the
    ``STATE`` component
    """

    EPOCH_PARTIALS = 2
    """
    Epoch partials; An N-element vector of the time-evolving partial derivatives
    of the propagated state w.r.t. the initial epoch where N is the size of the
    ``STATE`` component
    """

    PARAM_PARTIALS = 3
    """
    Parameter partials; An NxM matrix of the time-evolving partial derivatives
    of the propagated state w.r.t. the parameter values where N is the size of
    the ``STATE`` component and ``M`` is the number of parameters.

    Parameters are constant through an integration, i.e., they do not have their
    own governing differential equations. Parameters can include thrust magnitude,
    solar pressure coefficients, etc.
    """


class AbstractDynamicsModel(ABC):
    """
    Contains the mathematics that define a dynamical model

    Args:
        bodies ([Body]): one or more primary bodies
        properties: keyword arguments that define model properties

    Attributes:
        bodies (tuple): a tuple of :class:`~pika.data.Body` objects
        properties (dict): the model properties; these are constant across all
            integrations; e.g., a mass ratio for the CR3BP, or the initial phasing
            of multiple bodies in a four-body problem.
        charL (float): a characteristic length (km) used to nondimensionalize lengths
        charT (float): a characteristic time (sec) used to nondimensionalize times
        charM (float): a characteristic mass (kg) used to nondimensionalize masses
    """

    def __init__(self, *bodies, **properties):
        if any([not isinstance(body, Body) for body in bodies]):
            raise TypeError("Expecting Body objects")

        # Copy body objects into tuple
        self.bodies = tuple(copy(body) for body in bodies)

        # Unpack parameters into internal dict
        self._properties = {**properties}

        # Define default characteristic quantities as unity
        self._charL = 1.0  # km
        self._charT = 1.0  # sec
        self._charM = 1.0  # kg

    @property
    def properties(self):
        return copy(self._properties)

    @property
    def charL(self):
        return self._charL

    @property
    def charT(self):
        return self._charT

    @property
    def charM(self):
        return self._charM

    def __eq__(self, other):
        """
        Compare two Model objects. This can be overridden for more specific
        comparisons in derived classes
        """
        if not isinstance(other, AbstractDynamicsModel):
            return False

        if not type(self) == type(other):
            return False

        if not all([b1 == b2 for b1, b2 in zip(self.bodies, other.bodies)]):
            return False

        # TODO need to compare dicts by value??
        return (
            self.properties == other.properties
            and self.charL == other.charL
            and self.charT == other.charT
            and self.charM == other.charM
        )

    @abstractmethod
    def bodyState(self, ix, t, params):
        """
        Evaluate a body state vector at a time

        Args:
            ix (int): index of the body within :attr:`bodies`
            t (float): time value
            params (float, [float]): one or more parameter values

        Returns:
            numpy.ndarray: state vector for the body
        """
        pass

    @abstractmethod
    def diffEqs(self, t, y, varGroups, params):
        """
        Evaluate the differential equations that govern the variable array

        Args:
            t (float): time value
            y (numpy.ndarray<float>): One-dimensional variable array
            varGroups (tuple of VarGroups): describes the variable groups included
                in the ``y`` vector
            params (float, [float]): one or more parameter values. These are
                generally constants that may vary integration to integration
                within a model (e.g., thrust magnitude) but are not constants
                of the model itself (e.g., mass ratio).

        Returns:
            numpy.ndarray: the time derivative of the ``y`` vector
        """
        pass

    @property
    @abstractmethod
    def epochIndependent(self):
        """
        Returns:
            bool: True if the dynamics model has no dependencies on epoch, False
            otherwise.
        """
        pass

    @abstractmethod
    def stateSize(self, varGroups):
        """
        Get the size (i.e., number of elements) for one or more variable groups.

        Args:
            varGroups (VarGroups, [VarGroups]): describes one or more groups of variables

        Returns:
            int: the size of a variable array with the specified variable groups
        """
        pass

    def checkPartials(self, y0, tspan, params=None, initStep=1e-4, tol=1e-6):
        """
        This method logs information about the accuracy of the partials via the
        "pika.dynamics" logger. Erroneous partial values are logged at ERROR
        while all others are logged at INFO.
        """
        from pika import numerics
        from pika.propagate import Propagator

        allVars = [
            VarGroups.STATE,
            VarGroups.STM,
            VarGroups.EPOCH_PARTIALS,
            VarGroups.PARAM_PARTIALS,
        ]

        if not len(y0) == self.stateSize(allVars):
            raise ValueError(
                "y0 must define the full vector (STATE + STM + EPOCH_PARTIALS + PARAM_PARTIALS"
            )

        # TODO ensure tolerances are tight enough?
        prop = Propagator(self, dense=False)
        state0 = self.extractVars(y0, VarGroups.STATE, varGroupsIn=allVars)

        solution = prop.propagate(y0, tspan, params=params, varGroups=allVars)
        sol_vec = np.concatenate(
            [
                self.extractVars(solution.y[:, -1], grp, varGroupsIn=allVars).flatten()
                for grp in allVars[1:]
            ]
        )

        # Compute state partials (STM)
        def prop_state(y):
            sol = prop.propagate(y, tspan, params=params, varGroups=VarGroups.STATE)
            return sol.y[:, -1]

        num_stm = numerics.derivative_multivar(prop_state, state0, initStep)

        # Compute epoch partials
        if self.stateSize(VarGroups.EPOCH_PARTIALS) > 0:

            def prop_epoch(epoch):
                sol = prop.propagate(
                    state0,
                    [epoch + t for t in tspan],
                    params=params,
                    varGroups=VarGroups.STATE,
                )
                return sol.y[:, -1]

            num_epochPartials = numerics.derivative_multivar(
                prop_epoch, tspan[0], initStep
            )
        else:
            num_epochPartials = np.array([])

        # Compute parameter partials
        if self.stateSize(VarGroups.PARAM_PARTIALS) > 0:

            def prop_params(p):
                sol = prop.propagate(state0, tspan, params=p, varGroups=VarGroups.STATE)
                return sol.y[:, -1]

            num_paramPartials = numerics.derivative_multivar(
                prop_params, params, initStep
            )
        else:
            num_paramPartials = np.array([])

        # Combine into flat vector
        num_vec = np.concatenate(
            (
                num_stm.flatten(),
                num_epochPartials.flatten(),
                num_paramPartials.flatten(),
            )
        )

        # Now compare
        absDiff = abs(num_vec - sol_vec)
        relDiff = absDiff.copy()
        equal = True
        varNames = np.concatenate([self.varNames(grp) for grp in allVars[1:]])
        for ix in range(sol_vec.size):
            # Compute relative difference for non-zero numeric values
            if abs(num_vec[ix]) > 1e-12:
                relDiff[ix] = absDiff[ix] / abs(num_vec[ix])

            if abs(relDiff[ix]) > tol:
                equal = False
                logger.error(
                    f"Partial error in {ix+len(state0):02d} ({varNames[ix]}) is too large: "
                    f"Expected = {num_vec[ix]:.4e}, Actual = {sol_vec[ix]:.4e} "
                    f"(Rel err = {relDiff[ix]:.4e} > {tol:.2e})"
                )
            else:
                logger.info(
                    f"Partial error in {ix+len(state0):02d} ({varNames[ix]}) is ok: "
                    f"Expected = {num_vec[ix]:.4e}, Actual = {sol_vec[ix]:.4e} "
                    f"(Rel err = {relDiff[ix]:.4e}  <= {tol:.2e})"
                )

        return equal

    def extractVars(self, y, varGroups, varGroupsIn=None):
        """
        Extract a variable group from a vector

        Args:
            y (numpy.ndarray): the state vector
            varGroups (VarGroups): the variable group to extract
            varGroupsIn ([VarGroups]): the variable groups in ``y``. If ``None``, it
                is assumed that all variable groups with lower indices than
                ``varGroups`` are included in ``y``.

        Returns:
            numpy.ndarray: the subset of ``y`` that corresponds to the ``varGroups``
            group. The vector elements are reshaped into a matrix if applicable.

        Raises:
            ValueError: if ``y`` doesn't have enough elements to extract the
                requested variable groups
        """
        if varGroupsIn is None:
            varGroupsIn = [v for v in range(varGroups + 1)]
        varGroupsIn = np.array(varGroupsIn, ndmin=1)

        if not varGroups in varGroupsIn:
            raise RuntimeError(
                f"Requested variable group {varGroups} is not part of input set, {varGroupsIn}"
            )

        nPre = sum([self.stateSize(tp) for tp in varGroupsIn if tp < varGroups])
        sz = self.stateSize(varGroups)

        if y.size < nPre + sz:
            raise ValueError(
                f"Need {nPre + sz} vector elements to extract {varGroups} "
                f"but y has size {y.size}"
            )

        nState = self.stateSize(VarGroups.STATE)
        nCol = int(sz / nState)
        if nCol > 1:
            return np.reshape(y[nPre : nPre + sz], (nCol, nState))
        else:
            return np.array(y[nPre : nPre + sz])

    def defaultICs(self, varGroups):
        """
        Get the default initial conditions for a set of equations. This basic
        implementation returns a flattened identity matrix for the :attr:`~VarGroups.STM`
        and zeros for the other equation types. Derived classes can override
        this method to provide other values.

        Args:
            varGroups (VarGroups): describes the group of variables

        Returns:
            numpy.ndarray: initial conditions for the specified equation type
        """
        if varGroups == VarGroups.STM:
            return np.identity(self.stateSize(VarGroups.STATE)).flatten()
        else:
            return np.zeros((self.stateSize(varGroups),))

    def appendICs(self, y0, varsToAppend):
        """
        Append initial conditions for the specified variable groups to the
        provided state vector

        Args:
            y0 (numpy.ndarray): variable vector of arbitrary length
            varsToAppend (VarGroups): the variable group(s) to append initial
                conditions for.

        Returns:
            numpy.ndarray: an initial condition vector, duplicating ``q`` at
            the start of the array with the additional initial conditions
            appended afterward
        """
        y0 = np.asarray(y0)
        varsToAppend = np.array(varsToAppend, ndmin=1)
        nIn = y0.size
        nOut = self.stateSize(varsToAppend)
        y0_out = np.zeros((nIn + nOut,))
        y0_out[:nIn] = y0
        ix = nIn
        for v in sorted(varsToAppend):
            ic = self.defaultICs(v)
            if ic.size > 0:
                y0_out[ix : ix + ic.size] = ic
                ix += ic.size

        return y0_out

    def validForPropagation(self, varGroups):
        """
        Check that the set of variables can be propagated.

        In many cases, some groups of the variables are dependent upon others. E.g.,
        the STM equations of motion generally require the state variables to be
        propagated alongside the STM so ``VarGroups.STM`` would be an invalid set for
        evaluation but ``[VarGroups.STATE, VarGroups.STM]`` would be valid.

        Args:
            varGroups (VarGroups, [VarGroups]): the group(s) variables to be propagated

        Returns:
            bool: True if the set is valid, False otherwise
        """
        # General principle: STATE vars are always required
        return VarGroups.STATE in np.array(varGroups, ndmin=1)

    def varNames(self, varGroups):
        """
        Get names for the variables in each group.

        This implementation provides basic representations of the variables and
        should be overridden by derived classes to give more descriptive names.

        Args:
            varGroups (VarGroups): the variable group

        Returns:
            list of str: a list containing the names of the variables in the order
            they would appear in a variable vector.
        """
        N = self.stateSize(VarGroups.STATE)
        if varGroups == VarGroups.STATE:
            return [f"State {ix:d}" for ix in range(N)]
        elif varGroups == VarGroups.STM:
            return [f"STM({r:d},{c:d})" for r in range(N) for c in range(N)]
        elif varGroups == VarGroups.EPOCH_PARTIALS:
            return [
                f"Epoch Dep {ix:d}"
                for ix in range(self.stateSize(VarGroups.EPOCH_PARTIALS))
            ]
        elif varGroups == VarGroups.PARAM_PARTIALS:
            return [
                f"Param Dep({r:d},{c:d})"
                for r in range(N)
                for c in range(int(self.stateSize(VarGroups.PARAM_PARTIALS) / N))
            ]
        else:
            raise ValueError(f"Unrecognized enum: VarGroups = {varGroups}")

    def indexToVarName(self, ix, varGroups):
        # TODO test and document
        allNames = np.asarray(
            [self.varNames(varTp) for varTp in util.toList(varGroups)]
        ).flatten()
        return allNames[ix]


class ModelBlockCopyMixin:
    # A mixin class to prevent another class from copying stored DynamicsModels
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if isinstance(v, AbstractDynamicsModel):
                # Models should NOT be copied
                setattr(result, k, v)
            else:
                setattr(result, k, deepcopy(v, memo))
        return result
