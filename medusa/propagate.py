"""
Propagation
===========

Propagation, or numerical integration, is one of the core components of the 
library. The :func:`scipy.integrate.solve_ivp` function does the bulk of the
heavy lifting, with the :class:`Propagator` class providing more convenient 
data storage and functions.

.. autosummary:: Propagator

Events
------

It is often useful to be able to stop a propagation at a pre-specified event.
The :class:`AbstractEvent` class provides a generic interface for event
definitions with a variety of common event types defined.

.. autosummary::
   AbstractEvent
   ApseEvent
   BodyDistanceEvent
   DistanceEvent
   VariableValueEvent


Reference
==========

.. autoclass:: Propagator
   :members:

.. autoclass:: AbstractEvent
   :members:

.. autoclass:: ApseEvent
   :members:
   :show-inheritance:

.. autoclass:: BodyDistanceEvent
   :members:
   :show-inheritance:

.. autoclass:: DistanceEvent
   :members:
   :show-inheritance:

.. autoclass:: VariableValueEvent
   :members:
   :show-inheritance:

"""
import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import copy
from typing import Union

import numpy as np
from numba import njit  # type: ignore
from scipy.integrate import solve_ivp  # type: ignore
from scipy.optimize import OptimizeResult  # type: ignore

import medusa.util as util
from medusa.dynamics import AbstractDynamicsModel, ModelBlockCopyMixin, VarGroup
from medusa.typing import FloatArray, IntArray, override

logger = logging.getLogger(__name__)

# TODO benchmark prop speed with and without njit decorator; may not be adding
#   anything and just making code more complex


class Propagator(ModelBlockCopyMixin):
    """
    Create a propagator object.

    This class primarily wraps the :func:`scipy.integrate.solve_ivp` method

    Args:
        model: defines the dynamics model in which the propagation occurs
        method: the integration method to use; passed as the
            ``method`` argument to :func:`~scipy.integrate.solve_ivp`
        dense: whether or not to output dense propagation data;
            passed as the ``dense_output`` argument to
            :func:`~scipy.integrate.solve_ivp`
        atol: absolute tolerance for the integrator; see
            :func:`~scipy.integrate.solve_ivp` for more details
        rtol: relative tolerance for the integrator; see
            :func:`~scipy.integrate.solve_ivp` for more details
    """

    def __init__(
        self,
        model: AbstractDynamicsModel,
        method: str = "DOP853",
        dense: bool = True,
        atol: float = 1e-12,
        rtol: float = 1e-10,
    ) -> None:
        if not isinstance(model, AbstractDynamicsModel):
            raise ValueError("model must be derived from AbstractDynamicsModel")

        self.method: str = method  #: the numerical integration method
        self.atol: float = atol  #: absolute tolerance
        self.rtol: float = rtol  #: relative tolerance
        self.model: AbstractDynamicsModel = model  #: dynamics model
        self.dense: bool = dense  #: whether or not to output dense propagation data
        self.events: list[AbstractEvent] = []  #: integration events

    def propagate(
        self,
        w0: FloatArray,
        tspan: FloatArray,
        *,
        params: Union[FloatArray, None] = None,
        varGroups: Union[VarGroup, Sequence[VarGroup]] = VarGroup.STATE,
        **kwargs,
    ) -> OptimizeResult:
        """
        Perform a propagation

        Args:
            w0: initial state vector
            tspan: a 2-element vector defining the start and end times
                for the propagation
            params: parameters that are passed to the dynamics model
            varGroups: the variable groups that are included in ``w0``.
            kwargs: additional arguments passed to :func:`scipy.integrate.solve_ivp`.
                Note that the ``method`` and ``dense_output`` arguments, defined
                in the :class:`Propagator` constructor, cannot be overridden.

        Returns:
            scipy.optimize.OptimizeResult: the output of the
            :func:`~scipy.integrate.solve_ivp` function. This includes the times,
            states, event data, and other integration metadata.

        Raises:
            RuntimeError: if ``varGroups`` is not valid for the propagation as
                checked via :func:`AbstractDynamicsModel.validForPropagation`
            RuntimeError: if the size of ``w0`` is inconsistent with ``varGroups``
                as checked via :func:`AbstractDynamicsModel.groupSize`
            RuntimeError: if any of the objects in :attr:`events` are not derived
                from the :class:`AbstractEvent` base class
        """
        # Checks
        if not self.model.validForPropagation(varGroups):
            raise RuntimeError(f"VarGroup = {varGroups} is not valid for propagation")

        kwargs_in = {
            "method": self.method,
            "dense_output": self.dense,
            "atol": self.atol,
            "rtol": self.rtol,
        }
        kwargs.pop("method", None)
        kwargs.pop("dense_output", None)
        kwargs_in.update(**kwargs)

        if "args" in kwargs_in:
            logger.warning("Overwriting 'args' passed to propagate()")

        # make varGroups an array and then cast to tuple; need simple type for
        #   JIT compilation support
        varGroups = tuple(sorted(np.array(varGroups, ndmin=1)))

        # Ensure state is an array with the right number of elements
        w0_ = np.array(w0, ndmin=1, dtype=float, copy=True)
        if not w0_.size == self.model.groupSize(varGroups):
            # Likely scenario: user wants default ICs for other variable groups
            if w0_.size == self.model.groupSize(VarGroup.STATE):
                w0_ = self.model.appendICs(
                    w0_, [v for v in varGroups if not v == VarGroup.STATE]
                )
            else:
                raise RuntimeError(
                    f"w0 size is {w0_.size}, which is not the STATE size "
                    f"({self.model.groupSize(VarGroup.STATE)}); don't know how to respond."
                    " Please pass in a STATE-sized vector or a vector with all "
                    "initial conditions defined for the specified VarGroup"
                )

        kwargs_in["args"] = (varGroups, tuple(params) if params is not None else params)

        # Gather event functions and assign attributes
        for event in self.events:
            if not isinstance(event, AbstractEvent):
                raise RuntimeError(f"Event is not derived from AbstractEvent:\n{event}")
            event.assignEvalAttr()

        eventFcns = [event.eval for event in self.events]
        eventFcns.extend(util.toList(kwargs_in.get("events", [])))
        kwargs_in["events"] = eventFcns

        # Run propagation
        sol = solve_ivp(self.model.diffEqs, tspan, w0_, **kwargs_in)

        # Append our own metadata
        sol.model = self.model
        sol.params = copy(params)
        sol.varGroups = varGroups

        return sol


class AbstractEvent(ABC):
    """
    Defines an abstract event object

    Args:
        terminal: defines how the event interacts with the
            proapgation. If ``True``, the propagation will stop at the first occurrence
            of this event. If an :class:`int` is passed, the propagation will
            end after the specified number of occurrences.
        direction: defines event direction. If less than zero,
            the event is only triggered when ``eval`` moves from positive to
            negative values. A positive value triggers in the opposite direction,
            and ``0`` will trigger the event in either direction.
    """

    def __init__(self, terminal: bool = False, direction: float = 0.0) -> None:
        if not isinstance(terminal, bool):
            try:
                terminal = int(terminal)
            except Exception:
                raise TypeError("terminal input must be boolean or a number")

        try:
            direction = float(direction)
        except Exception:
            raise TypeError("direction input must be a number")

        if isinstance(terminal, int) and terminal < 0:
            logger.warning("Unexpected negative value for 'terminal'")

        self.terminal = terminal  #: bool: whether or not to terminate propagation
        self.direction = direction  #: float: event direction

    def assignEvalAttr(self) -> None:
        """
        Must be called before starting propagation so that the
        :attr:`terminal` and :attr:`direction` values take effect
        """
        self.eval.__func__.terminal = self.terminal  # type: ignore
        self.eval.__func__.direction = self.direction  # type: ignore

    @abstractmethod
    def eval(
        self,
        t: float,
        w: Sequence[float],
        varGroups: tuple[VarGroup, ...],
        params: Sequence[float],
    ) -> float:
        """
        Event evaluation method

        Args:
            t: independent variable value
            w: variable vector
            varGroups: describes the variable groups in ``w``
            params: extra parameters passed from the integrator

        Returns:
            the event value. The event occurs when this value is zero
        """
        pass


class ApseEvent(AbstractEvent):
    """
    Occurs at an apsis relative to a body in the model

    Args:
        model: a dynamics model
        bodyIx: index of the body
        terminal: defines how the event interacts with the
            proapgation. If ``True``, the propagation will stop at the first occurrence
            of this event. If an :class:`int` is passed, the propagation will
            end after the specified number of occurrences.
        direction: defines event direction. A positive value
            corresponds to periapsis, a negative value corresponds to an
            apoapsis, and a zero value finds all apses.
    """

    def __init__(
        self,
        model: AbstractDynamicsModel,
        bodyIx: int,
        terminal: bool = False,
        direction: float = 0.0,
    ) -> None:
        if not isinstance(model, AbstractDynamicsModel):
            raise ValueError("model must be derived from AbstractDynamicsModel")

        super().__init__(terminal, direction)
        self._model = model
        self._ix = bodyIx

    @override
    def eval(self, t, w, varGroups, params) -> float:
        return self._eval(t, w, tuple(varGroups), params, self._model, self._ix)

    @staticmethod
    # @njit  # would need to pass bodyState in as arg (TODO?)
    def _eval(
        t: float,
        w: FloatArray,
        varGroups: tuple[VarGroup, ...],
        params: FloatArray,
        model: AbstractDynamicsModel,
        ix: int,
    ) -> float:
        # apse occurs where dot product between primary-relative position and
        # velocity is zero
        # TODO this assumes the state vector is the Cartesian position and velocity
        #   vectors
        relState = w[:6] - model.bodyState(ix, t, w, varGroups, params)
        return (
            relState[0] * relState[3]
            + relState[1] * relState[4]
            + relState[2] * relState[5]
        )


class BodyDistanceEvent(AbstractEvent):
    """
    Occurs when the position distance relative to a body equals a specified value

    Args:
        model: the model that includes the body
        ix: index of the body in the model
        dist: distance value
        terminal: defines how the event interacts with the
            proapgation. If ``True``, the propagation will stop at the first occurrence
            of this event. If an :class:`int` is passed, the propagation will
            end after the specified number of occurrences.
        direction: defines event direction. If less than zero,
            the event is only triggered when ``eval`` moves from positive to
            negative values. A positive value triggers in the opposite direction,
            and ``0`` will trigger the event in either direction.
    """

    def __init__(
        self,
        model: AbstractDynamicsModel,
        ix: int,
        dist: float,
        terminal: bool = False,
        direction: float = 0.0,
    ) -> None:
        if not isinstance(model, AbstractDynamicsModel):
            raise ValueError("model must be derived from AbstractDynamicsModel")

        super().__init__(terminal, direction)
        self._model = model
        self._ix = ix
        self._dist = dist * dist  # evaluation uses squared value

    @override
    def eval(self, t, w, varGroups, params) -> float:
        # TODO assumes Cartesian position vector is the first piece of state vector
        relPos = w[:3] - self._model.bodyState(self._ix, t, w, varGroups, params)[:3]
        return sum([x * x for x in relPos]) - self._dist


class DistanceEvent(AbstractEvent):
    """
    Occurs when the L2 norm difference between two vectors equals the specified distance

    Args:
        dist: the distance
        vec: the goal vector
        ixs: the indices of the variables to compare to ``vec``
        terminal: defines how the event interacts with the
            proapgation. If ``True``, the propagation will stop at the first occurrence
            of this event. If an :class:`int` is passed, the propagation will
            end after the specified number of occurrences.
        direction: defines event direction. If less than zero,
            the event is only triggered when ``eval`` moves from positive to
            negative values. A positive value triggers in the opposite direction,
            and ``0`` will trigger the event in either direction.
    """

    def __init__(
        self,
        dist: float,
        vec: FloatArray,
        ixs: IntArray = [0, 1, 2],
        terminal: bool = False,
        direction: float = 0.0,
    ) -> None:
        vec, ixs = np.array(vec, ndmin=1), np.array(ixs, ndmin=1)
        if not vec.shape == ixs.shape:
            raise ValueError("vec and ixs must have the same shape")

        super().__init__(terminal, direction)
        self._vec = vec
        self._ixs = ixs
        self._dist = dist * dist  # evaluation uses squared value

    @override
    def eval(self, t, w, varGroups, params) -> float:
        dist = self._vec - w[self._ixs]
        return sum([x * x for x in dist]) - self._dist


class VariableValueEvent(AbstractEvent):
    """
    Occurs where a variable equals a specified value

    Args:
        varIx: index of the variable within the array
        varValue: the value of the variable
        terminal: defines how the event interacts with the
            proapgation. If ``True``, the propagation will stop at the first occurrence
            of this event. If an :class:`int` is passed, the propagation will
            end after the specified number of occurrences.
        direction: defines event direction. If less than zero,
            the event is only triggered when ``eval`` moves from positive to
            negative values. A positive value triggers in the opposite direction,
            and ``0`` will trigger the event in either direction.
    """

    def __init__(
        self,
        varIx: int,
        varValue: float,
        terminal: bool = False,
        direction: float = 0.0,
    ) -> None:
        super().__init__(terminal, direction)
        self._ix = varIx
        self._val = varValue

    @override
    def eval(self, t, w, varGroups, params) -> float:
        return self._eval(w, self._ix, self._val)

    @staticmethod
    @njit
    def _eval(w: FloatArray, ix: int, val: float) -> float:
        return float(val - w[ix])
