"""
Dynamical Models
================

Within the context of the ``medusa`` library, a dynamical model defines a set of
differential equations that describe the motion of a celestial object such as a
planet, moon, asteroid, star, or spacecraft. This library generally assumes that
the object's **state** is described by its Cartesian position and velocity coordinates,

.. math::
    \\vec{q}(\\tau, T, \\vec{p}) = 
    \\begin{Bmatrix} x & y & z & \dot{x} & \dot{y} & \dot{z} \end{Bmatrix}^T,

where :math:`\\vec{q}` is the state vector, :math:`\\tau` is the independent variable
(usually time), :math:`T` is an absolute epoch corresponding to :math:`\\tau = 0`, 
and :math:`\\vec{p}` is a set of constant parameters (e.g., engine thrust).
In this notation, the dot over the coordinates represents the derivative with 
respect to :math:`\\tau`. Some models may append more 
variables, such as a thrust vector parameterization, to this state vector.

In addition to the state vector, models can include equations that describe three
other "variable groups", described by :class:`VarGroup`, that provide extra 
dynamical information and are frequently
used in differential corrections and optimization processes:

- :data:`VarGroup.STM`, the state transition matrix, :math:`\mathbf{\Phi}`. 
  This matrix relates changes 
  in :math:`\\vec{q}` at one value of :math:`\\tau` to changes in :math:`\\vec{q}`
  at a different value, i.e.,

  .. math::
     \delta \\vec{q}(\\tau_2) = \mathbf{\Phi}(\\tau_1, \\tau_2) \delta \\vec{q}(\\tau_1)

  When :math:`\\tau_2 = \\tau_1`, the state transition matrix is identity.

- :data:`VarGroup.EPOCH_PARTIALS`, epoch partials, :math:`\partial \\vec{q} / \partial T`. 
  A vector of partial derivatives of the state vector with respect to the epoch

- :data:`VarGroup.PARAM_PARTIALS`, parameter partials, :math:`\partial \\vec{q} / \partial \\vec{q}`. 
  A matrix of partial derivatives of the state vector with respect to the parameter vector


Grouped together with matrices expanded in row-major order, the variable groups
are collected into a **variable vector**, denoted by

.. math::
   \\vec{w} = \\begin{Bmatrix} 
     \\vec{q} & \\mathbf{\Phi} & 
     \\frac{\partial \\vec{q}}{\partial T} & 
     \\frac{\partial \\vec{q}}{\partial \\vec{p}}
   \\end{Bmatrix}.

The :class:`AbstractDynamicsModel` defines the evolution of this vector
via the :func:`~AbstractDynamicsModel.diffEqs` function, which returns the
deriative of the variable vector with respect to the independent variable,

.. math::
   \dot{\\vec{w}} = \\frac{\mathrm{d} \\vec{w}}{\mathrm{d} \\tau}(\\tau, T, \\vec{p})

Several other methods are supplied to initialize, extract, and append variable
groups:

.. autosummary::
   :nosignatures:

   ~AbstractDynamicsModel.groupSize
   ~AbstractDynamicsModel.defaultICs
   ~AbstractDynamicsModel.appendICs
   ~AbstractDynamicsModel.extractGroup
   ~AbstractDynamicsModel.varNames
   ~AbstractDynamicsModel.validForPropagation

Additional information about the system of equations are available via properties
and attributes of the model,

.. autosummary::
   :nosignatures:

   ~AbstractDynamicsModel.charL
   ~AbstractDynamicsModel.charT
   ~AbstractDynamicsModel.charM
   ~AbstractDynamicsModel.isEpochIndependent

Finally, a convenience method to check the partial derivatives included in the
differential equations is provided. This method is primarily for debugging purposes
but can save many hours of time when deriving and coding the partial differential
equations.

.. autosummary::
   :nosignatures:

   ~AbstractDynamicsModel.checkPartials

Implementations
---------------

The ``AbstractDynamicsModel`` defined in this module is just that -- abstract.
Concrete implementations are included in the submodules listed below.

.. toctree::
   :maxdepth: 1

   dynamics.crtbp
   dynamics.lowthrust

Reference
-----------------

.. autoclass:: VarGroup
   :members:
   :show-inheritance:

.. autoclass:: AbstractDynamicsModel
   :members:

"""
import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from copy import copy, deepcopy
from enum import IntEnum
from typing import Any, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from medusa import util
from medusa.data import Body
from medusa.typing import FloatArray

logger = logging.getLogger(__name__)

__all__ = [
    # base module
    "VarGroup",
    "AbstractDynamicsModel",
    "ModelBlockCopyMixin",
    # sub modules
    "crtbp",
    "lowthrust",
]


# numba JIT compilation only supports Enum and IntEnum
class VarGroup(IntEnum):
    """
    Specify the variable groups included in a model variable array. The integer
    values of the groups correspond to their location in a variable array. I.e.,
    the ``STATE`` variables are always first, followed by the ``STM``,
    ``EPOCH_PARTIALS``, and ``PARAM_PARTIALS``. All matrix objects are stored in
    row-major order within the variable vector.
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
        bodies: one or more primary bodies
        properties: keyword arguments that define model properties
    """

    def __init__(
        self,
        *bodies: Body,
        **properties: Any,
    ) -> None:
        if any([not isinstance(body, Body) for body in bodies]):
            raise TypeError("Expecting Body objects")

        # Copy body objects into tuple
        #: tuple[Body]: the bodies
        self.bodies = tuple(copy(body) for body in bodies)

        # Unpack parameters into internal dict
        self._properties = {**properties}

        # Define default characteristic quantities as unity
        self._charL = 1.0  # km
        self._charT = 1.0  # sec
        self._charM = 1.0  # kg

    @property
    def properties(self) -> dict:
        """
        The model properties. These are constant across all integrations
        """
        return copy(self._properties)

    @property
    def charL(self) -> float:
        """A characteristic length (km) used to nondimensionalize lengths"""
        return self._charL

    @property
    def charT(self) -> float:
        """A characteristic time (sec) used to nondimensionalize times"""
        return self._charT

    @property
    def charM(self) -> float:
        """A characteristic mass (kg) used to nondimensionalize masses"""
        return self._charM

    def __eq__(self, other: object) -> bool:
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
    def bodyState(
        self,
        ix: int,
        t: float,
        w: FloatArray,
        varGroups: tuple[VarGroup, ...],
        params: FloatArray,
    ) -> NDArray[np.double]:
        """
        Evaluate a body state vector

        Args:
            ix: index of the body within :attr:`bodies`
            t: independent variable (e.g., time)
            w: variable vector
            varGroups: describes the variable groups included in the ``w`` vector
            params: one or more parameter values

        Returns:
            the state vector for the body
        """
        pass

    @abstractmethod
    def diffEqs(
        self,
        t: float,
        w: FloatArray,
        varGroups: tuple[VarGroup, ...],
        params: FloatArray,
    ) -> NDArray[np.double]:
        """
        Evaluate the differential equations that govern the variable array

        Args:
            t: independent variable (e.g., time)
            w: variable vector
            varGroups: describes the variable groups included
                in the ``w`` vector
            params: one or more parameter values. These are
                generally constants that may vary integration to integration
                within a model (e.g., thrust magnitude) but are not constants
                of the model itself (e.g., mass ratio).

        Returns:
            the derivative of the ``w`` vector with respect to ``t``
        """
        pass

    @property
    @abstractmethod
    def epochIndependent(self) -> bool:
        """
        Returns:
            True if the differetial equations have no dependencies on epoch,
            False otherwise.
        """
        pass

    @abstractmethod
    def groupSize(self, varGroups: Union[VarGroup, Sequence[VarGroup]]) -> int:
        """
        Get the size (i.e., number of elements) for one or more variable groups.

        Args:
            varGroups: describes one or more groups of variables

        Returns:
            the size of a variable array with the specified variable groups
        """
        pass

    def checkPartials(
        self,
        w0: FloatArray,
        tspan: FloatArray,
        params: Union[FloatArray, None] = None,
        initStep: float = 1e-4,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        printTable: bool = True,
    ) -> bool:
        """
        Check the partial derivatives included in the differential equations

        Args:
            w0: a full state vector (includes all VarGroup) for this model
            tspan: a 2-element vector defining the start and end times
                for numerical propagation
            params: propagation parameters
            initStep: the initial step size for the multivariate
                numerical derivative function in :func:`numerics.derivative_multivar`
            rtol: the numeric and analytical values are
                equal when the absolute value of (numeric - analytic)/numeric
                is less than ``rtol``
            atol: the numeric and analytic values are equal
                when the absolute value of (numeric - analytic) is less than ``atol``
            printTable: whether or not to print a table of the
                partial derivatives, their expected (numeric) and actual (analytic)
                values, and the relative and absolute errors between the expected
                and actual values.

        Returns:
            True if each partial derivative satisfies the relative *or*
            absolute tolerance; False is returned if any of the partials fail
            both tolerances.
        """
        from rich.table import Table

        from medusa import console, numerics
        from medusa.propagate import Propagator

        allVars = [
            VarGroup.STATE,
            VarGroup.STM,
            VarGroup.EPOCH_PARTIALS,
            VarGroup.PARAM_PARTIALS,
        ]

        if not len(w0) == self.groupSize(allVars):
            raise ValueError(
                "w0 must define the full vector (STATE + STM + EPOCH_PARTIALS + PARAM_PARTIALS"
            )

        # TODO ensure tolerances are tight enough?
        prop = Propagator(self, dense=False)
        w0 = np.array(w0, copy=True)
        state0 = self.extractGroup(w0, VarGroup.STATE, varGroupsIn=allVars)

        solution = prop.propagate(w0, tspan, params=params, varGroups=allVars)
        sol_vec = np.concatenate(
            [
                self.extractGroup(solution.y[:, -1], grp, varGroupsIn=allVars).flatten()
                for grp in allVars[1:]
            ]
        )

        # Compute state partials (STM)
        def prop_state(y):
            sol = prop.propagate(y, tspan, params=params, varGroups=VarGroup.STATE)
            return sol.y[:, -1]

        num_stm = numerics.derivative_multivar(prop_state, state0, initStep)

        # Compute epoch partials
        if self.groupSize(VarGroup.EPOCH_PARTIALS) > 0:

            def prop_epoch(epoch):
                sol = prop.propagate(
                    state0,
                    [epoch + t for t in tspan],
                    params=params,
                    varGroups=VarGroup.STATE,
                )
                return sol.y[:, -1]

            num_epochPartials = numerics.derivative_multivar(
                prop_epoch, tspan[0], initStep  # type: ignore
            )
        else:
            num_epochPartials = np.array([])

        # Compute parameter partials
        if self.groupSize(VarGroup.PARAM_PARTIALS) > 0:

            def prop_params(p):
                sol = prop.propagate(state0, tspan, params=p, varGroups=VarGroup.STATE)
                return sol.y[:, -1]

            num_paramPartials = numerics.derivative_multivar(
                prop_params, params, initStep  # type: ignore
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
        table = Table(
            "Status",
            "Name",
            "Numeric",
            "Analytic",
            "Rel Err",
            "Abs Err",
            title="Partial Derivative Check",
        )

        for ix in range(sol_vec.size):
            # Compute relative difference for non-zero numeric values
            if abs(num_vec[ix]) > 1e-12:
                relDiff[ix] = absDiff[ix] / abs(num_vec[ix])

            relOk = abs(relDiff[ix]) <= rtol
            rStyle = "i" if relOk else "u"
            absOk = abs(absDiff[ix]) <= atol
            aStyle = "i" if absOk else "u"

            table.add_row(
                "OK" if relOk or absOk else "ERR",
                varNames[ix],
                f"{num_vec[ix]:.4e}",
                f"{sol_vec[ix]:.4e}",
                f"[{rStyle}]{relDiff[ix]:.4e}[/{rStyle}]",
                f"[{aStyle}]{absDiff[ix]:.4e}[/{aStyle}]",
                style="blue" if relOk or absOk else "red",
            )

            if not (relOk or absOk):
                equal = False

        if printTable:
            console.print(table)

        return equal

    def extractGroup(
        self,
        w: NDArray[np.double],
        varGroup: VarGroup,
        varGroupsIn: Union[Sequence[VarGroup], None] = None,
    ) -> NDArray[np.double]:
        """
        Extract a variable group from a variable vector

        Args:
            w: the state vector
            varGroup: the variable group to extract
            varGroupsIn: the variable groups in ``w``. If ``None``, it
                is assumed that all variable groups with lower indices than
                ``varGroup`` are included in ``w``.

        Returns:
            the subset of ``w`` that corresponds to the ``VarGroup``
            group. The vector elements are reshaped into a matrix if applicable.

        Raises:
            ValueError: if ``w`` doesn't have enough elements to extract the
                requested variable groups
        """
        if varGroupsIn is None:
            varGroupsIn = [VarGroup(v) for v in range(varGroup + 1)]
        # varGroupsIn = np.array(varGroupsIn, ndmin=1)

        if not varGroup in varGroupsIn:
            raise RuntimeError(
                f"Requested variable group {varGroup} is not part of input set, {varGroupsIn}"
            )

        nPre = sum([self.groupSize(tp) for tp in varGroupsIn if tp < varGroup])
        sz = self.groupSize(varGroup)

        if w.size < nPre + sz:
            raise ValueError(
                f"Need {nPre + sz} vector elements to extract {varGroup} "
                f"but w has size {w.size}"
            )

        nState = self.groupSize(VarGroup.STATE)
        nCol = int(sz / nState)
        if nCol > 1:
            return np.reshape(w[nPre : nPre + sz], (nState, nCol))
        else:
            return np.array(w[nPre : nPre + sz])

    def defaultICs(self, varGroup: VarGroup) -> NDArray[np.double]:
        """
        Get the default initial conditions for a variable vector.

        This basic implementation returns a flattened identity matrix for the
        :attr:`~VarGroup.STM` and zeros for the other equation types. Derived
        classes can override this method to provide different values.

        Args:
            varGroup: describes the group of variables

        Returns:
            initial conditions for the specified equation type
        """
        if varGroup == VarGroup.STM:
            return np.identity(self.groupSize(VarGroup.STATE)).flatten()
        else:
            return np.zeros((self.groupSize(varGroup),))

    def appendICs(
        self, w0: ArrayLike, varsToAppend: Union[VarGroup, Sequence[VarGroup]]
    ) -> NDArray[np.double]:
        """
        Append initial conditions for the specified variable groups to the
        provided variable vector

        Args:
            w0: variable vector of arbitrary length
            varsToAppend: the variable group(s) to append initial conditions for.

        Returns:
            an initial condition vector, duplicating ``w0`` at
            the start of the array with the additional initial conditions
            appended afterward
        """
        w0_ = np.asarray(w0, copy=True)
        nIn = w0_.size
        nOut = self.groupSize(varsToAppend)
        w0_out = np.zeros((nIn + nOut,))
        w0_out[:nIn] = w0_
        ix = nIn
        for v in sorted(util.toList(varsToAppend)):
            ic = self.defaultICs(v)
            if ic.size > 0:
                w0_out[ix : ix + ic.size] = ic
                ix += ic.size

        return w0_out

    def validForPropagation(
        self, varGroups: Union[VarGroup, Sequence[VarGroup]]
    ) -> bool:
        """
        Check that the set of variables can be propagated.

        In many cases, some groups of the variables are dependent upon others. E.g.,
        the STM equations of motion generally require the state variables to be
        propagated alongside the STM so ``VarGroup.STM`` would be an invalid set for
        evaluation but ``[VarGroup.STATE, VarGroup.STM]`` would be valid.

        Args:
            varGroups: the group(s) variables to be propagated

        Returns:
            True if the set is valid, False otherwise
        """
        # General principle: STATE vars are always required
        return VarGroup.STATE in np.array(varGroups, ndmin=1)

    def varNames(self, varGroup: VarGroup) -> list[str]:
        """
        Get names for the variables in each group.

        This implementation provides basic representations of the variables and
        should be overridden by derived classes to give more descriptive names.

        Args:
            varGroup (VarGroup): the variable group

        Returns:
            list of str: a list containing the names of the variables in the order
            they would appear in a variable vector.
        """
        N = self.groupSize(VarGroup.STATE)
        if varGroup == VarGroup.STATE:
            return [f"State {ix:d}" for ix in range(N)]
        elif varGroup == VarGroup.STM:
            return [f"STM({r:d},{c:d})" for r in range(N) for c in range(N)]
        elif varGroup == VarGroup.EPOCH_PARTIALS:
            return [
                f"Epoch Dep {ix:d}"
                for ix in range(self.groupSize(VarGroup.EPOCH_PARTIALS))
            ]
        elif varGroup == VarGroup.PARAM_PARTIALS:
            return [
                f"Param Dep({r:d},{c:d})"
                for r in range(N)
                for c in range(int(self.groupSize(VarGroup.PARAM_PARTIALS) / N))
            ]
        else:
            raise ValueError(f"Unrecognized enum: varGroup = {varGroup}")

    def indexToVarName(self, ix: int, varGroups: Sequence[VarGroup]) -> str:
        # TODO test and document
        allNames = np.asarray(
            [self.varNames(varTp) for varTp in util.toList(varGroups)]
        ).flatten()
        return allNames[ix]


class ModelBlockCopyMixin:
    """
    A mixin class that prevents the parent class from copying stored
    :class:`AbstractDynamisModel` objects
    """

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
