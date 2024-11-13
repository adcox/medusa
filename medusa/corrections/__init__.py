"""
Differential Corrections
========================

This module provides objects to define the differential corrections problem,
its constraints, and solver objects.

Overview
------------------

At the simplest level, a differential corrections problem is composed of **variables**
and **constraints**. The goal of the corrections process is to adjust the variables
until all of the constraints are satisfied. Variables, a.k.a.., "free variables"
"design variables" can include quantities like a state vector, a time-of-flight,
an epoch, or a parameter.

.. autosummary::
   :nosignatures:

   Variable
   AbstractConstraint

Mathematically, these free variables are collected into a single free variable vector,
:math:`\\vec{X}`. Similarly, the constraints are collected into a constraint
vector that is a function of the free variables, :math:`\\vec{F}(\\vec{X})`. 
The problem is solved when :math:`\\vec{F}(\\vec{X}_f) = \\vec{0}`, with the solution
represented by the :math:`\\vec{X}_f` variables.

.. autosummary::
   :nosignatures:

   CorrectionsProblem.freeVarVec
   CorrectionsProblem.constraintVec

An iterative Newton method is used to solve corrections problems. Given an initial
set of variables, :math:`\\vec{X}_*`, the constraint function can be expanded 
about :math:`\\vec{X}_*` in a Taylor Series expansion:

.. math::
   \\vec{F}(\\vec{X}_f) = \\vec{F}(\\vec{X}_*) + \\mathbf{J} (\\vec{X}_f - \\vec{X}_*) + \\mathrm{h.o.t.s}

where :math:`\\mathbf{J}` is the **Jacobian** matrix with elements
:math:`\\mathbf{J}_{i,j} = \\partial \\vec{F}_i / \\partial \\vec{X}_j`, evaluated
at :math:`\\vec{X}_*`. 

.. autosummary:: CorrectionsProblem.jacobian
   :nosignatures:

Ignoring the higher-order terms (h.o.t.s) and recognizing
that :math:`\\vec{F}(\\vec{X}_f)` evaluates to zero, the expansion is reduced to

.. math::
   \\vec{F}(\\vec{X}_*) + \\mathbf{J}(\\vec{X}_f - \\vec{X}_*) = \\vec{0}.

In practice, because the higher order terms have been ignored, this gradient-based
method requires multiple iterations to locate :math:`\\vec{X}_f`. The iterative
**update equation** is written as

.. math::
   -\\vec{F}(\\vec{X}_n) = \\mathbf{J}(\\vec{X}_{n+1} - \\vec{X}_n).


Defining a Problem
------------------

Variables
^^^^^^^^^

Variables within a corrections problem are managed via a few functions:

.. autosummary::
   :nosignatures:

   CorrectionsProblem.addVariables
   CorrectionsProblem.rmVariables
   CorrectionsProblem.clearVariables

Variables can be added and removed from a problem in any order, including intermixed
operations with the methods to add/remove constraints.


Constraints
^^^^^^^^^^^

Constraints within a corrections problem are managed via a few functions:

.. autosummary::
   :nosignatures:

   CorrectionsProblem.addConstraints
   CorrectionsProblem.rmConstraints
   CorrectionsProblem.clearConstraints

Like variables, constraints can be added and removed from a problem in any order.

Trajectory-Aware Corrections
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Within the context of multi-body dynamics, variables are usually grouped together
to describe epoch-states and propagated arcs.
Control point and segment objects provide a way to group related variables together.
A **control point** groups a state vector and an epoch together.
The control point also defines a dynamical model in which the state and epoch
are defined (with implications for the central body, frame, and units).

Similarly, a **segment** stores variables about a propagation between two points:
the time-of-flight and propagation parameters. Each segment is linked to an
"origin" node that represents the initial epoch-state and defines the dynamical
model for the propagation.

.. autosummary::
   :nosignatures:

   ControlPoint
   Segment

However they are grouped and stored, variables must be added to a corrections problem.
The :class:`ShootingProblem` offers convenience methods to incorporate segments without
explicitly adding all of the stored variables (states, epochs, times-of-flight, 
etc.). This problem type also performs checks to ensure the segments combine 
into a valid graph.
Constraints are added to the problem via the same interface as in the more general
:class:`CorrectionsProblem`.

.. autosummary::
   :nosignatures:

   ShootingProblem.addSegments
   ShootingProblem.rmSegments
   ShootingProblem.checkValidGraph

The ``ShootingProblem`` requires a post-processing step after all of the
segments and constraints have been added. The :func:`ShootingProblem.build` 
function performs the operations to import variables from the ``Segment`` and
``ControlPoint`` objects.

.. autosummary:: ShootingProblem.build
   :nosignatures:

Debugging
^^^^^^^^^^

Maintaining a mental map of the variables and constraints and their relationships
to each other can be difficult to do. A few methods provide helpful information:

.. autosummary::
   :nosignatures:

   CorrectionsProblem.printFreeVars
   CorrectionsProblem.printConstraints
   CorrectionsProblem.printJacobian
   CorrectionsProblem.checkJacobian


Constraints
-----------

A library of constraints is included in the ``corrections.constraints`` submodule,

.. toctree::
   :maxdepth: 1

   corrections.constraints

While these constraints cover many common needs for differential corrections, 
new types and formulations of constraints are always a necessity. To facilitate
user-defined constraints, the :class:`AbstractConstraint` interface is provided.
Derived objects must define the constraint function and provide partial derivatives
of the constraint function with respect to relevant variables. The ``CorrectionsProblem``
object provides a function to compare analytical and numerical derivatives from 
constraints, a helpful tool when debugging the derivation of a new constraint.

.. autosummary:: CorrectionsProblem.checkJacobian
   :nosignatures:

Solving Problems
----------------

An iterative differential corrections process is used to update the variables
until the constraints are satisfied, i.e., :math:`\\vec{F}(\\vec{X}) = \\vec{0}`.
A :class:`DifferentialCorrector` performs this iterative process with 
configurable options for the state update and the convergence check.

.. autosummary:: DifferentialCorrector.solve
   :nosignatures:

State Update
^^^^^^^^^^^^

The **state update** is the process of solving the equation

.. math::
   -\\vec{F}(\\vec{X}_n) = \\mathbf{J}(\\vec{X}_{n+1} - \\vec{X}_n) = \\mathbf{J}\\delta \\vec{X}_n

for the update, or "step", :math:`\\delta \\vec{X}_n`. When the the number of free
variables is greater than or equal to the number of constraints, :math:`\\mathbf{J}`
is square (equal) or wide (greater than) and the update equation can be solved
via a minimum norm update. When the number of constraints is greater than the 
number of variables, a least squares update can be applied. It is up to the user
to specify which update to apply; this is accomplished by setting the
:attr:`DifferentialCorrector.updateGenerator` attribute. Two options are currently
available,

.. autosummary::
   :nosignatures:

   MinimumNormUpdate
   LeastSquaresUpdate

.. warning:: The update generator API is rough and likely to be updated in the future.

Convergence Check
^^^^^^^^^^^^^^^^^

Mathematically, the corrections problem is solved when :math:`\\vec{F}(\\vec{X}) = \\vec{0}`.
Of course, in practice the vector will never evaluate *exactly* to zero, so some
numerical tolerance is required. There are also several ways to quantify how close
the vector is to zero, including several types of vector norms. One option is 
currently available, and can be set via the 
:attr:`DifferentialCorrector.convergeCheck` attribute.

.. autosummary:: L2NormConvergence
   :nosignatures:

.. warning:: The convergence check API is rough and likely to be updated in the future.

Reference
==============

.. autoclass:: AbstractConstraint
   :members:

.. autoclass:: CorrectionsProblem
   :members:

.. autoclass:: ControlPoint
   :members:

.. autoclass:: DifferentialCorrector
   :members:

.. autoclass:: Segment
   :members:

.. autoclass:: Variable
   :members:

.. autoclass:: MinimumNormUpdate
   :members:

.. autoclass:: LeastSquaresUpdate
   :members:

.. autoclass:: L2NormConvergence
   :members:

.. autoclass:: ShootingProblem
   :members:

"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import copy, deepcopy
from typing import Union

import numpy as np
import numpy.ma as ma
import scipy
from numpy.typing import NDArray
from rich.columns import Columns
from rich.panel import Panel
from rich.table import Table
from scipy.integrate._ivp.ivp import OdeResult

from medusa import console, numerics, util
from medusa.dynamics import DynamicsModel, ModelBlockCopyMixin, State, VarGroup
from medusa.propagate import Propagator
from medusa.typing import FloatArray

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
    "L2NormConvergence",
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


class Variable(ma.MaskedArray):
    """
    Contains a variable vector with an optional mask to flag non-variable values.

    This class is derived from :class:`~numpy.ma.MaskedArray`, adding a ``name``
    attribute and a few convenience methods. Data elements that are unmasked
    (mask = False) are termed "free variables."

    Args:
        data: scalar or array of variable values
        mask: ``True`` flags values as excluded from the free variable vector;
            ``False`` flags values as included. If a single value is provided,
            all ``values`` will have that mask applied.
        name: a name for the variable

    Like regular numpy masked arrays, variables can be constructed a variety of
    ways.

    .. code-block:: python

       # Only the final three values are "free"
       state = Variable([1,2,3,4,5,6], mask=[1,1,1,0,0,0])

       # A time-of-flight value that is not free
       tof = Variable(1.23, mask=True, name="time-of-flight")
    """

    # SEE https://numpy.org/doc/stable/user/basics.subclassing.html

    def __new__(cls, data, mask=False, name="", **kwargs):
        # Create the MaskedArray instance of our type, given the usual input args.
        # This will call the standard constructor but return an object of our type.
        # It also triggers a call to __array_finalize__

        # Create the masked array; force scalar data into an array
        obj = ma.MaskedArray(np.array(data, ndmin=1), mask=mask, **kwargs).view(cls)
        # obj = super().__new__(cls, np.array(data, ndmin=1), mask=mask, **kwargs)
        obj.name = name
        return obj

    def __init__(
        self,
        data: Union[float, FloatArray],
        mask: Union[bool, Sequence[bool]] = False,
        name: str = "",
        **kwargs,
    ):
        # Called after __new__ and __array_finalize__. Because those functions
        #   have done all of the initialization, this one exists mostly for
        #   documentation purposes
        super().__init__()

        self.name = name  #: the variable name

    def __array_finalize__(self, obj) -> None:
        # self is a new object resulting from the
        #   MaskedArray.__new__(Variable, ...) call, therefore it
        #   only has attributes that the MaskedArray.__new__() call gave it.
        super().__array_finalize__(obj)

        # We could have gotten to the MaskedArray.__new__ call in 3 ways:
        #
        # 1) from an explicit constructor, e.g., Variable()
        #       In this case, ``obj`` is None because we're in the middle of the
        #       Variable.__new__ constructor
        if obj is None:
            return

        # 2) from view casting, e.g., arr.view(Variable)
        #       In this case, ``obj`` is arr and type(obj) may be Variable
        # 3) from new-from-template, e.g., variable[:3]
        #       In this case, type(obj) is Variable
        self.name = getattr(obj, "name", "")

    def __eq__(var1, var2):
        # Redefine equality to return a scalar boolean so we can search for
        # a variable object in a list, for example. Use the ID so that equality
        # is only true if the objects occupy the same space in memory. Otherwise
        # comparisons like (var in dict) will return false positives for variables
        # with equal values but different objects.
        return id(var1) == id(var2)

    def __hash__(self):
        # Define hash so Variables can be used as keys in dicts. Do NOT use
        #  data or mask in hash because they are mutable
        return hash(id(self))

    def __repr__(self):
        out = "<Variable"
        if self.name:
            out += " {!r}".format(self.name)
        out += ": {!s}>".format(super().__str__())
        return out

    @property
    def freeVals(self) -> NDArray[np.double]:
        """Only free, i.e., unmasked, values"""
        return self.data[~self.mask]

    @property
    def numFree(self) -> int:
        """The number of free, i.e., unmasked, values"""
        return int(sum(~self.mask))


class AbstractConstraint(ModelBlockCopyMixin, ABC):
    """
    Defines the interface for a constraint object
    """

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        """Default to-string implementation"""
        return f"<{self.__class__.__name__}>"

    @property
    @abstractmethod
    def size(self) -> int:
        """
        Get the number of scalar constraint equations

        Returns:
            the number of constraint equations
        """
        pass

    def clearCache(self) -> None:
        """
        Called by correctors when variables are updated to signal a constraint
        that any caches ought to be cleared and re-computed.

        The default implementation of this method does nothing; derived classes
        should implement it if they cache data.
        """
        pass

    @abstractmethod
    def evaluate(self) -> NDArray[np.double]:
        """
        Evaluate the constraint

        Returns:
            the value of the constraint funection; evaluates
            to zero when the constraint is satisfied

        Raises:
            RuntimeError: if the constraint cannot be evaluated
        """
        pass

    @abstractmethod
    def partials(
        self, freeVarIndexMap: dict[Variable, int]
    ) -> dict[Variable, NDArray[np.double]]:
        """
        Compute the partial derivatives of the constraint vector with respect to
        all variables

        Args:
            freeVarIndexMap: maps the first index (:class:`int`) of each
                :class:`Variable` within ``freeVars`` to the variable object

        Returns:
            a dictionary mapping a :class:`Variable` object to the partial
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
        state: the state at which to create the control point. The state
            :func:`~State.values` and :attr:`~State.time` are used to construct
            the ``stateVec`` and ``epoch`` variables. If you wish to pass in
            variables explicitly, use the :func:`fromVars` method.
        autoMask: whether or not to auto-mask the ``epoch`` variable.
            If True and ``model``
            :func:`~medusa.dynamics.DynamicsModel.epochIndependent` is True,
            the ``epoch`` variable has its mask set to True.

    Raises:
        TypeError: if ``state`` is not derived from :class:`State`
    """

    def __init__(self, state, autoMask=True):
        if not isinstance(state, State):
            raise TypeError("state must be derived from medusa.dynamics.State")

        # TODO document that stateVec and state will have different values...

        # Define attributes
        self.model = state.model  #: DynamicsModel: the associated model
        self.epoch = Variable([])  #: Variable: the epoch
        self.stateVec = Variable([])  #: Variable: the state vector

        self.state = copy(state)
        self._makeVars(state.time, state.get(VarGroup.STATE), autoMask)

    @staticmethod
    def fromProp(
        solution: OdeResult, ix: int = 0, autoMask: bool = True
    ) -> ControlPoint:
        """
        Construct a control point from a propagated arc

        Args:
            solution: the output from a propagation
            ix: the index of the point within the ``solution``
            autoMask: whether or not to auto-mask the ``epoch``
                variable. If True and ``solution.model``
                :func:`~medusa.dynamics.DynamicsModel.epochIndependent`
                is True, the ``epoch`` variable has its mask set to True.

        Returns:
            a control point with epoch and state retrieved from the
            propagation solution

        Raises:
            ValueError: if ``ix`` is out of bounds with respect to the ``solution`` indices
        """
        if not isinstance(solution, OdeResult):
            raise TypeError("Expecting OdeResult from scipy solve_ivp")

        if ix > len(solution.t):
            raise ValueError(f"ix = {ix} is out of bounds (max = {len(solution.t)})")

        return ControlPoint(solution.states[ix], autoMask)

    @staticmethod
    def fromVars(
        model: DynamicsModel,
        epoch: Union[float, Variable],
        stateVec: Union[FloatArray, Variable],
        center: str,
        frame: str,
        autoMask: bool = True,
    ):
        """
        Construct a control point from variables

        Args:
            model: defines the dynamics model for the propagation
            epoch: the epoch at which the propagation begins. An input ``float``
                is converted to a :class:`Variable` with name "Epoch"
            state: the state at which the propagation begins. An input list of
                floats is converted to a :class:`Variable` with name "State"
            autoMask: whether or not to auto-mask the ``epoch`` variable.
                If True and ``model``
                :func:`~medusa.dynamics.DynamicsModel.epochIndependent` is True,
                the ``epoch`` variable has its mask set to True.

        Raises:
            TypeError: if the model is not derived from DynamicsModel
            RuntimeError: if the epoch specifies more than one value
            RuntimeError: if the state specifies more values than the dynamics model
                allows for :attr:`VarGroup.STATE`
        """
        if not isinstance(model, DynamicsModel):
            raise TypeError("Model must be derived from DynamicsModel")

        # Create a control point
        q = stateVec.data if isinstance(stateVec, Variable) else stateVec
        t = epoch.data if isinstance(epoch, Variable) else epoch
        stateObj = model.makeState(q, t, center, frame)
        cp = ControlPoint(stateObj)

        # Overwrite the variables using definitions passed in
        cp._makeVars(epoch, stateVec, autoMask)
        return cp

    def _makeVars(
        self,
        epoch: Union[float, Variable],
        stateVec: Union[FloatArray, Variable],
        autoMask: bool,
    ):
        """
        Args:
            model: defines the dynamics model for the propagation
            epoch: the epoch at which the propagation begins. An input ``float``
                is converted to a :class:`Variable` with name "Epoch"
            state: the state at which the propagation begins. An input list of
                floats is converted to a :class:`Variable` with name "State"
            autoMask: whether or not to auto-mask the ``epoch`` variable.
                If True and ``model``
                :func:`~medusa.dynamics.DynamicsModel.epochIndependent` is True,
                the ``epoch`` variable has its mask set to True.

        Raises:
            RuntimeError: if the epoch specifies more than one value
            RuntimeError: if the state specifies more values than the dynamics model
                allows for :attr:`VarGroup.STATE`
        """
        if not isinstance(epoch, Variable):
            epoch = Variable(epoch, name="Epoch")

        if not isinstance(stateVec, Variable):
            stateVec = Variable(stateVec, name="State")

        if not epoch.size == 1:
            raise RuntimeError("Epoch can only have one value")

        sz = self.state.groupSize(VarGroup.STATE)
        if not stateVec.size == sz:
            raise RuntimeError("State must have {sz} values")

        if autoMask:
            epoch.mask = self.model.epochIndependent

        self.epoch = epoch
        self.stateVec = stateVec

    def __repr__(self) -> str:
        out = "<ControlPoint:"
        out += f"\n  model=<{self.model.__class__.__module__}.{self.model.__class__.__name__}>,"
        for attr in ("epoch", "stateVec", "state"):
            out += "\n  {!s}={!r}".format(attr, getattr(self, attr))
        out += "\n>"
        return out

    @property
    def importableVars(self) -> tuple[Variable, ...]:
        """
        A tuple of variables that can be imported into a corrections problem.

        Importing is accomplished via :func:`CorrectionsProblem.importVariables`.
        These variables are defined as a property so that evaluation occurs at
        runtime, allowing the user to modify or reassign the variables.
        """
        return self.stateVec, self.epoch


class Segment:
    """
    Defines a numerical propagation starting from an origin control point for a
    specified time-of-flight.

    Args:
        origin: the starting point for the propagated trajectory.
            This control point defines the dynamics model, the initial state, and
            the initial epoch.
        tof: the time-of-flight for the propagated trajectory
            in units consistent with the origin's model and state. If the input
            is a float, a :class:`Variable` object with name "Time-of-flight"
            is created.
        terminus: a control point located at the terminus
            of the propagated arc. This is useful when defining differential
            corrections problems that seek to constrain the numerical integration.
            It is not required that the terminus state or epoch are numerically
            consistent with the state or epoch values at the end of the arc. The
            ``terminus`` is not used or modified by the Segment.
        prop: the propagator to use. If ``None``, a
            :class:`~medusa.propagate.Propagator` is constructed for the ``origin``
            model with the ``dense_output`` flag set to False to reduce computational
            load.
        propParams: parameters to be
            passed to the model's equations of motion. If the input is not a
            :class:`Variable`, a variable is constructed with a name "Params"
    """

    def __init__(
        self,
        origin: ControlPoint,
        tof: Union[float, Variable],
        terminus: Union[None, ControlPoint] = None,
        prop: Union[None, Propagator] = None,
        propParams: Union[FloatArray, Variable] = [],
    ) -> None:
        if not isinstance(origin, ControlPoint):
            raise TypeError("origin must be a ControlPoint")

        if terminus is not None and not isinstance(terminus, ControlPoint):
            raise TypeError("terminus must be a ControlPoint or None")

        if prop is not None:
            if not isinstance(prop, Propagator):
                raise TypeError("prop must be a Propagator")
            prop = copy(prop)
        else:
            prop = Propagator(dense_output=False)

        if not isinstance(tof, Variable):
            tof = Variable(tof, name="Time-of-flight")
        else:
            if not tof.size == 1:
                raise RuntimeError("Time-of-flight variable must define only one value")

        if not isinstance(propParams, Variable):
            propParams = Variable(propParams, name="Params")

        #: ContrlolPoint: the starting point for the propagated trajectory.
        #: Defines the dynamics model, initial state, and initial epoch.
        self.origin = origin

        #: Union[None, ControlPoint]: the stoping point for the propagated trajectory
        #: Does not affect the propagation.
        self.terminus = terminus

        #: Variable: the time-of-flight
        self.tof = tof

        #: Propagator: the propagator
        self.prop = prop

        #: Variable: propagation parameters
        self.propParams = propParams

        #: Union[None, OdeResult]: the propagation output, if
        #: :func:`propagate` has been called, otherwise ``None``.
        self.propSol = None

    def __repr__(self) -> str:
        out = "<Segment:"
        for attr in ("origin", "terminus", "tof"):
            out += "\n  {!s}={!r},".format(attr, getattr(self, attr))
        out += "\n>"
        return out

    @property
    def importableVars(self) -> tuple[Variable, ...]:
        """
        A tuple of variables that can be imported into a corrections problem.

        Importing is accomplished via :func:`CorrectionsProblem.importVariables`.
        These variables are defined as a property so that evaluation occurs at
        runtime, allowing the user to modify or reassign the variables.
        """
        return self.tof, self.propParams

    def resetProp(self) -> None:
        """
        Reset the propagation cache, :attr:`propSol`, to ``None``
        """
        self.propSol = None

    def state(self, ix: int = -1) -> NDArray[np.double]:
        """
        Lazily propagate the trajectory and retrieve the state

        Args:
            ix: the index within the :attr:`propSol` ``y`` array.

        Returns:
            the propagated state (:class:`VarGroup` ``STATE``) on the
            propagated trajectory
        """
        sol = self.propagate(VarGroup.STATE)
        return sol.states[ix].get(VarGroup.STATE)

    def partials_state_wrt_time(self, ix: int = -1) -> NDArray[np.double]:
        """
        Lazily propagate the trajectory and retrieve the partial derivatives

        Args:
            ix: the index within the :attr:`propSol` ``y`` array.

        Returns:
            the partials of the propagated state with respect to
            :attr:`tof`, i.e., the time derivative of the
            :attr:`~medusa.dynamics.VarGroup.STATE` variables
        """
        sol = self.propagate(VarGroup.STATE)
        dy_dt = self.origin.model.diffEqs(
            sol.t[ix], sol.y[:, ix], (VarGroup.STATE,), tuple(self.propParams.data)
        )
        return self.origin.state._extractGroup(VarGroup.STATE, vals=dy_dt)

    def partials_state_wrt_initialState(self, ix: int = -1) -> NDArray[np.double]:
        """
        Lazily propagate the trajectory and retrieve the partial derivatives

        Args:
            ix: the index within the :attr:`propSol` ``y`` array.

        Returns:
            the partials of the propagated state with respect to the
            :attr:`origin` ``state``, i.e., the :attr:`~medusa.dynamics.VarGroup.STM`.
            The partials are returned in matrix form.
        """
        sol = self.propagate([VarGroup.STATE, VarGroup.STM])
        return sol.states[ix].get(VarGroup.STM)

    def partials_state_wrt_epoch(self, ix: int = -1) -> NDArray[np.double]:
        """
        Lazily propagate the trajectory and retrieve the partial derivatives

        Args:
            ix: the index within the :attr:`propSol` ``y`` array.

        Returns:
            the partials of the propagated state with respect to
            the :attr:`origin` ``epoch``, i.e., the
            :attr:`~medusa.dynamics.VarGroup.EPOCH_PARTIALS`.
        """
        sol = self.propagate([VarGroup.STATE, VarGroup.STM, VarGroup.EPOCH_PARTIALS])
        partials = sol.states[ix].get(VarGroup.EPOCH_PARTIALS)

        # Handle models that don't depend on epoch by setting partials to zero
        if partials.size == 0:
            partials = np.zeros((self.origin.state.groupSize(VarGroup.STATE),))

        return partials

    def partials_state_wrt_params(self, ix: int = -1) -> NDArray[np.double]:
        """
        Lazily propagate the trajectory and retrieve the partial derivatives

        Args:
            ix: the index within the :attr:`propSol` ``y`` array.

        Returns:
            the partials of the propagated state with respect to
            the :attr:`propParams`, i.e., the :attr:`~medusa.dynamics.VarGroup.PARAM_PARTIALS`.
        """
        sol = self.propagate(
            [
                VarGroup.STATE,
                VarGroup.STM,
                VarGroup.EPOCH_PARTIALS,
                VarGroup.PARAM_PARTIALS,
            ]
        )
        partials = sol.states[ix].get(VarGroup.PARAM_PARTIALS)

        # Handle models that don't depend on propagator params by setting partials
        # to zero
        if partials.size == 0:
            partials = np.zeros((self.origin.state.groupSize(VarGroup.STATE),))

        return partials

    def propagate(
        self,
        varGroups: Union[VarGroup, Sequence[VarGroup]],
        lazy: bool = True,
        **kwargs,
    ) -> OdeResult:
        """
        Propagate from the :attr:`origin` for the specified :attr:`tof`. Results
        are stored in :attr:`propSol`.

        Args:
            varGroups: which equations of motion should be included
                in the propagation.
            lazy: whether or not to lazily propagate the
                trajectory. If True, the :attr:`prop` ``propagate()`` function is
                called if :attr:`propSol` is ``None`` or if the previous propagation
                did not include one or more of the ``varGroups``.
            kwargs: additional arguments passed to the
                :func:`medusa.propagate.Propagator.propagate` function. These
                arguments do not affect the lazy evaluation. For instance, if the
                propagation has previously run with ``dense_output = False``,
                the propagation is *not* rerun when ``dense_output = True`` is
                an input; set ``lazy = False`` to force a new propagation.

        Returns:
            the propagation result. This is also stored in :attr:`propSol`.
        """
        # Check to see if we can skip the propagation
        if lazy and self.propSol is not None:
            varGroups = np.array(varGroups, ndmin=1)
            if all([v in self.propSol.varGroups for v in varGroups]):
                return self.propSol

        # Propagate from the origin for TOF
        tspan = [0, self.tof.data[0]] + self.origin.epoch.data[0]
        q0 = self.origin.model.makeState(
            self.origin.stateVec.data,
            self.origin.epoch.data[0],
            self.origin.state.center,
            self.origin.state.frame,
        )

        # Save the solution
        self.propSol = self.prop.propagate(
            q0,
            tspan,
            params=self.propParams.data,
            varGroups=varGroups,
            **kwargs,
        )

        return self.propSol

    def denseEval(self, num: int = 100) -> OdeResult:
        """
        Evaluate the propagated solution with a dense grid of times.

        This is often useful before plotting the solution.

        .. note::
           If the segment has not been propagated at all (:attr:`propSol` is ``None``),
           the origin state will be propagated via :func:`propagate` with the
           :data:`~medusa.dynamics.VarGroup.STATE` variable group.

        Args:
            num: the number of points between the :attr:`origin` Epoch and that
                epoch plus the :attr:`tof`. Times are spaced linearly.

        Returns:
            The densely evaluated solution with updated ``t`` and ``y`` attributes.
        """
        times = np.linspace(
            self.origin.epoch.data[0],
            self.origin.epoch.data[0] + self.tof.data[0],
            num=num,
        )
        if self.propSol is None:
            self.propSol = self.propagate(VarGroup.STATE, dense_output=True)

        self.propSol = self.prop.denseEval(self.propSol, times)
        return self.propSol


# ------------------------------------------------------------------------------
# Organizing it all together
# ------------------------------------------------------------------------------


class CorrectionsProblem:
    """
    Defines a mathematical problem to be solved by a corrections algorithm
    """

    def __init__(self) -> None:
        # free variables and constraints are stored in a list for add/remove
        self._freeVars: list = []
        self._constraints: list = []

        # Other data objects are initialized to None and recomputed on demand
        self._freeVarIndexMap: Union[None, dict[Variable, int]] = None
        self._constraintIndexMap: Union[None, dict[AbstractConstraint, int]] = None

        self._freeVarVec: Union[None, NDArray[np.double]] = None
        self._constraintVec: Union[None, NDArray[np.double]] = None
        self._jacobian: Union[None, NDArray[np.double]] = None

    def __repr__(self) -> str:
        out = f"<{self.__class__.__name__}:"

        out += "\n  {!s} free variables".format(len(self._freeVars))
        if self._freeVarIndexMap is not None:
            out += ":"
            for var, ix in self._freeVarIndexMap.items():
                out += "\n    [{!s}, {!s}): {!s},".format(
                    ix, ix + var.numFree, var.name or "(unnamed)"
                )
        else:
            out += ","

        out += "\n  {!s} constraints".format(len(self._constraints))
        if self._constraintIndexMap is not None:
            out += ":"
            for con, ix in self._constraintIndexMap.items():
                out += "\n    [{!s}, {!s}): {!s},".format(
                    ix, ix + con.size, con.__class__.__name__
                )
        else:
            out += ","

        for attr in (
            "freeVarIndexMap",
            "constraintIndexMap",
            "freeVarVec",
            "constraintVec",
            "jacobian",
        ):
            val = getattr(self, "_" + attr)
            out += "\n  {!s} = {!s},".format(attr, "None" if val is None else "Cached")

        out += "\n>"
        return out

    @staticmethod
    def fromIteration(
        problem: CorrectionsProblem, correctorLog: dict, it: int = -1
    ) -> CorrectionsProblem:
        """
        Create a corrections problem from a logged corrector iteration

        Args:
            problem: a template corrections problem that
                matches the free-variable and constraint structure of the
                logged solutions. This can be either the initial guess or the
                converged solution.
            correctorLog: the corrections log output from
                :func:`DifferentialCorrector.solve`
            it: the iteration index

        Returns:
            A corrections problem that matches the specified iteration.
        """
        prob = deepcopy(problem)

        freevars = correctorLog["iterations"][it]["free-vars"]
        prob.updateFreeVars(freevars)

        return prob

    # -------------------------------------------
    # Variables

    def addVariables(self, variable: Union[Variable, Sequence[Variable]]) -> None:
        """
        Add one or more variables to the problem

        This clears the caches for :func:`freeVarIndexMap`, :func:`freeVarVec`,
        and :func:`jacobian`.

        Args:
            variable: one or more variables to add. Variables are stored by
                reference (they are not copied).

        Raises:
            ValueError: if any of the inputs are not :class:`Variable` objects
        """
        if isinstance(variable, Variable):
            variable = [variable]

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

    def importVariables(self, obj: object) -> None:
        """
        Imports variables from an object. The object must define a ``importableVars``
        attribute that contains any variables that can be imported by the problem.
        Variables are added via :func:`addVariable`, clearing the caches if
        applicable.

        Args:
            obj: an object that defines an ``importableVars`` attribute.
        """
        self.addVariables(getattr(obj, "importableVars", []))

    def rmVariables(self, variable: Union[Variable, Sequence[Variable]]) -> None:
        """
        Remove one or more variables from the problem

        This clears the caches for :func:`freeVarIndexMap`, :func:`freeVarVec`,
        and :func:`jacobian`.

        Args:
            variable: the variable(s) to remove. If the
                variable is not part of the problem, no action is taken.
        """
        if isinstance(variable, Variable):
            variable = [variable]

        for var in util.toList(variable):
            try:
                self._freeVars.remove(var)
                self._freeVarIndexMap = None
                self._freeVarVec = None
                self._jacobian = None
            except ValueError:
                logger.debug(f"Could not remove variable {var}")

    def clearVariables(self) -> None:
        """
        Remove all variables from the problem

        This clears the caches for :func:`freeVarIndexMap`, :func:`freeVarVec`,
        and :func:`jacobian`.
        """
        self._freeVars = []
        self._freeVarIndexMap = None
        self._freeVarVec = None
        self._jacobian = None

    def freeVarVec(self) -> NDArray[np.double]:
        """
        Get the free variable vector

        Returns:
            the free variable vector. This result is cached until the free
            variables are updated.
        """
        if self._freeVarVec is None:
            self._freeVarVec = np.zeros((self.numFreeVars,))
            for var, ix in self.freeVarIndexMap().items():
                self._freeVarVec[ix : ix + var.numFree] = var.freeVals

        return self._freeVarVec

    def freeVarIndexMap(self) -> dict[Variable, int]:
        """
        Get the free variable index map

        Returns:
            a dictionary mapping the variables included in the problem
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

    def freeVarIndexMap_reversed(self) -> dict[int, Variable]:
        # TODO document
        # TODO could a ChainMap work here?
        varMap = {}
        for var, ix0 in self.freeVarIndexMap().items():
            for k in range(var.numFree):
                assert (
                    not ix0 + k in varMap
                ), "duplicate entry in reversed freeVarIndexMap"
                varMap[ix0 + k] = var
        return varMap

    def updateFreeVars(self, newVec: NDArray[np.double]) -> None:
        """
        Update the free variable vector and corresponding Variable objects

        This clears the caches for :func:`freeVarVec`, :func:`constraintVec`,
        and :func:`jacobian`. The :func:`AbstractConstraint.clearCache` method is
        also called on all constraints in the problem.

        Args:
            newVec: an updated free variable vector. It must
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
            var.data[~var.mask] = newVec[ix0 : ix0 + var.numFree]

        self._freeVarVec = None
        self._constraintVec = None
        self._jacobian = None
        for constraint in self._constraints:
            constraint.clearCache()

    @property
    def numFreeVars(self) -> int:
        """
        Get the number of scalar free variable values in the problem. This is
        distinct from the number of :class:`Variable` objects as each object
        can contain a vector of values and a mask that removes some of the values
        from the free variable vector

        Returns:
            number of free variables
        """
        return sum([var.numFree for var in self._freeVars])

    # -------------------------------------------
    # Constraints

    def addConstraints(
        self, constraint: Union[AbstractConstraint, Sequence[AbstractConstraint]]
    ) -> None:
        """
        Add one or more constraints to the problem. If the constraint defines variables,
        they are also added to the problem via :func:`importVariables`.

        This clears the caches for :func:`constraintIndexMap`,
        :func:`constraintVec`, and :func:`jacobian`

        Args:
            constraint: one or more constraints to add. Constraints are stored by
                reference (they are not copied).
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

    def rmConstraints(
        self, constraint: Union[AbstractConstraint, Sequence[AbstractConstraint]]
    ) -> None:
        """
        Remove one or more constraints from the problem. If the constraint defines
        variables, they are also removed from the problem.

        This clears the caches for :func:`constraintIndexMap`,
        :func:`constraintVec`, and :func:`jacobian`

        Args:
            constraint: the constraint(s) to remove. If the constraint is not part
                of the problem, no action is taken.
        """
        for con in util.toList(constraint):
            try:
                self._constraints.remove(con)
                self.rmVariables(getattr(con, "importableVars", []))
                self._constraintIndexMap = None
                self._constraintVec = None
                self._jacobian = None
            except ValueError:
                logger.debug(f"Could not remove constraint {con}")

    def clearConstraints(self) -> None:
        """
        Remove all constraints from the problem.

        This clears the caches for :func:`constraintIndexMap`,
        :func:`constraintVec`, and :func:`jacobian`
        """
        self._constraints = []
        self._constraintIndexMap = None
        self._constraintVec = None
        self._jacobian = None

    def constraintVec(self) -> NDArray[np.double]:
        """
        Get the constraint vector

        Returns:
            the constraint vector. This result is cached
            until the free variables or constraints are updated.
        """
        if self._constraintVec is None:
            self._constraintVec = np.zeros((self.numConstraints,))
            for constraint, ix in self.constraintIndexMap().items():
                self._constraintVec[ix : ix + constraint.size] = constraint.evaluate()
        return self._constraintVec

    def constraintIndexMap(self) -> dict[AbstractConstraint, int]:
        """
        Get the constraint index map

        Returns:
            a dictionary mapping the constraints included in the problem
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

    def constraintIndexMap_reversed(self) -> dict[int, AbstractConstraint]:
        conMap = {}
        for con, ix0 in self.constraintIndexMap().items():
            for k in range(con.size):
                assert (
                    not ix0 + k in conMap
                ), "duplicate entry in reversed constraintIndexMap"
                conMap[ix0 + k] = con
        return conMap

    @property
    def numConstraints(self) -> int:
        """
        Get the number of scalar constraint equations. This is distinct from the
        number of :class:`AbstractConstraint` objects as each object can define
        a vector of equations.

        Returns:
            the number of scalar constraint equations
        """
        return sum([con.size for con in self._constraints])

    # -------------------------------------------
    # Jacobian

    def jacobian(
        self, numeric: bool = False, stepSize: float = 1e-4
    ) -> NDArray[np.double]:
        """
        Get the Jacobian matrix, i.e., the partial derivative of the constraint
        vector with respect to the free variable vector. The rows of the matrix
        correspond to the scalar constraints and the columns of the matrix
        correspond to the scalar free variables.

        Args:
            numeric: if False, the Jacobian is computed analytically using partial
                derivatives from the dynamics model and the constraints.
                Otherwise, the :func:`medusa.numerics.derivative_multivar` method is
                used to compute the Jacobian numerically.
            stepSize: if ``numeric`` is True, this is the step
                size for the ``derivative_multivar`` function.

        Returns:
            the Jacobian matrix. This result is cached until either
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
                    if any(~partialVar.mask):
                        cols = self.freeVarIndexMap()[partialVar] + np.arange(
                            partialVar.numFree
                        )
                        for rix, partials in zip(
                            cix + np.arange(constraint.size), partialMat
                        ):
                            if len(partialMat.shape) > 1:
                                self._jacobian[rix, cols] = partials[~partialVar.mask]
                            elif not partialVar.mask[0]:
                                self._jacobian[rix, cols] = partials

        return self._jacobian

    def checkJacobian(
        self,
        stepSize: float = 1e-4,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        printTable: bool = True,
    ) -> bool:
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
        ``(numeric - analytical)/numeric``. The two are deemed equal if the
        absolute difference is less than ``atol`` OR the relative difference is
        less than ``rtol``.

        This is not a foolproof method (numerics can be tricky) but is often a
        helpful tool in debugging partial derivative derivations.

        Args:
            stepSize: the initial free variable step size; this
                value should be sufficiently large such that the constraint vector
                changes *substantially* as a result.
            rtol: relative tolerance for equality between the
                numeric and analytical Jacobian matrices
            atol: absolute tolerance for equality between the
                numeric and analystical Jacobian matrics
            printTable: whether or not to plot a table with the
                comparison data to screen.

        Returns:
            True if the numeric and analytical Jacobian matrices are equal
        """
        from rich.table import Table

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
        table = Table(
            "Status",
            "Row",
            "Constraint",
            "Col",
            "Variable",
            "Numeric",
            "Analytical",
            "Rel Err",
            "Abs Err",
        )
        for r, row in enumerate(relDiff):
            for c in range(len(row)):
                # If numeric partial is nonzero, compute relative dif
                if abs(numeric[r, c]) > 1e-12:
                    relDiff[r, c] = absDiff[r, c] / numeric[r, c]

                relOk = abs(relDiff[r, c]) <= rtol
                rStyle = "i" if relOk else "u"
                absOk = abs(absDiff[r, c]) <= atol
                aStyle = "i" if absOk else "u"

                table.add_row(
                    "OK" if relOk or absOk else "ERR",
                    f"{r}",
                    conMap[r].__class__.__name__,
                    f"{c}",
                    varMap[c].name,
                    f"{numeric[r,c]:.4e}",
                    f"{analytic[r,c]:.4e}",
                    f"[{rStyle}]{relDiff[r,c]:.4e}[/{rStyle}]",
                    f"[{aStyle}]{absDiff[r,c]:.4e}[/{aStyle}]",
                    style="blue" if relOk or absOk else "red",
                )

                if not (relOk or absOk):
                    equal = False

        if printTable:
            console.print(table)

        return equal

    # -------------------------------------------
    # Printing
    def printFreeVars(self) -> None:
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

    def printConstraints(self) -> None:
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

    def printJacobian(self, numeric: bool = False) -> None:
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

    def __init__(self) -> None:
        super().__init__()

        self._segments: list = []
        self._points: list = []
        self._adjMat: Union[None, NDArray[np.int_]] = None  # adjacency matrix

    @property
    def controlPoints(self) -> list[ControlPoint]:
        # TODO unit test
        return self._points

    @property
    def segments(self) -> list[Segment]:
        # TODO unit test
        return self._segments

    def addSegments(self, segment: Union[Segment, Sequence[Segment]]) -> None:
        """
        Add one or more segments to the problem. Segments are not added if they
        already exist in the problem.

        Args:
            segment: the segment(s) to add to the problem

        Raises:
            ValueError: if one of the inputs is not a :class:`Segment`
        """
        for seg in util.toList(segment):
            if not isinstance(seg, Segment):
                raise ValueError("Can only add segment objects")

            if not seg in self._segments:
                self._segments.append(seg)
                self._adjMat = None

    def rmSegments(self, segment: Union[Segment, Sequence[Segment]]) -> None:
        """
        Remove one or more segments from the problem. Segments that are not
        in the problem are ignored.

        Args:
            segment: the segment(s) to remove from the problem
        """
        for seg in util.toList(segment):
            try:
                self._segments.remove(seg)
                self._adjMat = None
            except ValueError:
                logger.debug(f"Can not remove {seg} from segments")

    def build(self) -> None:
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

    def adjacencyMatrix(self) -> NDArray[np.int_]:
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
            the adjacency matrix. This value is cached and reset
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

    def checkValidGraph(self) -> list[str]:
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
                    tofSignSum += int(np.sign(tof.data)[0])
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
    """

    def __init__(self) -> None:
        #: int: maximum number of iterations that can be attempted
        self.maxIterations = 20

        # TODO define an interface?
        #: An object containing a ``isConverged`` method that accepts a
        #: :class`CorrectionsProblem` as an input and returns a :class:`bool`
        self.convergenceCheck = L2NormConvergence(1e-8)

        # TODO define an interface?
        #: An object containing and ``update`` method that accepts a
        #: :class:`CorrectionsProblem` as an input and returns a :class:`numpy.ndarray`
        #: representing the change in the free variable vector.
        self.updateGenerator = MinimumNormUpdate()

        #: A perisistent log; reset and populated every time :func:`solve` is run.
        self.log: dict[str, object] = {}

    def __repr__(self) -> str:
        out = f"<{self.__class__.__name__}:"
        for attr in ("convergenceCheck", "updateGenerator"):
            out += "\n  {!s} = {!r},".format(attr, getattr(self, attr))
        out += "\n>"
        return out

    def _validateArgs(self) -> None:
        """
        Check the types and values of class attributes so that useful errors
        can be thrown before those attributes are evaluated or used.

        Raises:
            TypeError: if any attribute type is incorrect
            ValueError: if any attribute value is invalid
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

    def solve(
        self, problem: CorrectionsProblem, lineSearch: bool = True
    ) -> tuple[CorrectionsProblem, dict]:
        """
        Solve a corrections problem by iterating the :attr:`updateGenerator`
        until the :attr:`convergenceCheck` returns True or the :attr:`maxIterations`
        are reached.

        Args:
            problem: the corrections problem to be solved

        Returns:
            A tuple with two elements. The first is a :class:`CorrectionsProblem`
            that stores the solution after the final iteration of the solver. The
            second is a logging :class:`dict` with the following keywords:

            - ``status`` (:class:`str`): the status of the solver after the final
              iteration. Can be "empty" if the problem contained no free variables
              or constraints, "converged" if the convergence check was satisfied,
              or "max-iterations" if the maximum number of iterations were completed
              before convergence.
            - ``iterations`` (:class:`list`): a list of `dict`; each dict represents
              an iteration of the solver and includes a copy of the free variable
              vector and the constraint vector.

            The log from the most recent call to ``solve`` is stored in :attr:`log`
            in case the solver encounters an error and does not return the log.
        """
        tolA = 1e-12  # tolerance for spurious convergence; TODO user set
        self._validateArgs()

        solution = deepcopy(problem)
        # TODO clear all caches in solution

        self.log = {"status": "", "iterations": [], "lineSearch": lineSearch}

        if solution.numFreeVars == 0 or solution.numConstraints == 0:
            self.log["status"] = "empty"
            return solution, copy(self.log)

        def costFunc(freeVarVec: NDArray[np.double]) -> float:
            # Function to minimize in line search
            solution.updateFreeVars(freeVarVec)
            F = solution.constraintVec()
            return float(0.5 * F.T @ F)

        with warnings.catch_warnings(record=True) as caught:
            warnings.filterwarnings("error", module="scipy.linalg")
            itCount = 0
            logger.info("Beginning differential corrections")
            while True:
                if itCount > 0:
                    freeVarStep = self.updateGenerator.update(solution)

                    if lineSearch:
                        # Get the current constraint vector
                        F = solution.constraintVec()

                        # Calculate the gradient of costFunc w.r.t. freeVars
                        grad = F.T @ solution.jacobian()

                        newVec, _, checkMin = numerics.linesearch(
                            costFunc,
                            solution.freeVarVec(),
                            float(0.5 * F.T @ F),
                            grad,
                            freeVarStep,
                        )

                        if checkMin:
                            # Check for gradient of costFunc = 0, i.e., spurious convergence.
                            # Algorithm from the `newt` function defined in Sec 9.7 of
                            #   Numerical Recipes in C, 2nd Edition; Press, Teukolsky,
                            #   Vetterling, and Flannery. 1996.
                            funcVal = costFunc(newVec)
                            F = solution.constraintVec()
                            denom = max(funcVal, 0.5 * newVec.size)
                            grad = F.T @ solution.jacobian()
                            test = max(
                                [
                                    abs(grad[ix]) * max(newVec[ix], 1.0) / denom
                                    for ix in range(newVec.size)
                                ]
                            )
                            if test < tolA:
                                raise RuntimeError(
                                    "The line search has converged to a local minimum"
                                    " of f = (1/2) FX * FX. Try another initial guess"
                                    " for the free variable vector"
                                )

                    else:
                        newVec = solution.freeVarVec() + freeVarStep

                    solution.updateFreeVars(newVec)

                self.log["iterations"].append(  # type: ignore
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
                    self.log["status"] = "converged"
                    break
                elif itCount >= self.maxIterations:
                    self.log["status"] = "max-iterations"
                    break

        return solution, copy(self.log)


class MinimumNormUpdate:
    """
    Computes the minimum-norm update.

    The update equation,

    .. math::
       -\\vec{F}(\\vec{X}_n) = \\mathbf{J}\\delta \\vec{X}_n

    can be solved via the minimum norm method when :math:`\\mathbf{J}` is square
    or when the number of columns (free variables) is greater than the number of
    rows (constraints).

    When :math:`\\mathbf{J}` is not square, an infinite number of solutions to this
    equation exist. As the name suggests, the "minimum norm" solution minimizes
    the L2 norm of the step. The resulting update equation is

    .. math::
       \\delta \\vec{X}_n = -\\mathbf{J}^T (\\mathbf{JJ}^T)^{-1} \\vec{F}(\\vec{X}_n)
    """

    def update(self, problem: CorrectionsProblem) -> NDArray[np.double]:
        """
        Compute the minimum-norm update

        Args:
            problem: the corrections problem to be solved

        Returns:
            the free variable vector step, `dX`
        """
        nFree = problem.numFreeVars
        # using -FX for all equations, so premultiply
        FX = -1 * problem.constraintVec()

        if len(FX) > nFree:
            raise RuntimeError(
                "Minimum Norm Update requires fewer or equal number of "
                "constraints as free variables"
            )
        else:
            jacobian = problem.jacobian()
            if len(FX) == nFree:
                # Jacobian is square; it must be full rank!
                # Solve the linear system J @ dX = -FX for dX
                return scipy.linalg.solve(jacobian, FX)
            else:  # len(FX) < nFree
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

    def update(self, problem: CorrectionsProblem) -> NDArray[np.double]:
        """
        Compute the least-squares update

        Args:
            problem: the corrections problem to be solved

        Returns:
            the free variable vector step, `dX`
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
        tol: the maximum L2 norm for convergence
    """

    def __init__(self, tol: float) -> None:
        #: float: the maximum L2 norm for convergence
        self.tol = tol

    def isConverged(self, problem: CorrectionsProblem) -> bool:
        """
        Args:
            problem: the corrections problem to be evaluated

        Returns:
            True if the L2 norm of the :func:`~CorrectionsProblem.constraintVec`
            is less than or equal to ``tol``
        """
        return bool(np.linalg.norm(problem.constraintVec()) <= self.tol)
