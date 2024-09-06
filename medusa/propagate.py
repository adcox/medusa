"""
Propagation, i.e., Numerical Integration
"""
import logging
from abc import ABC, abstractmethod
from copy import copy

import numpy as np
from numba import njit
from scipy.integrate import solve_ivp

from medusa.dynamics import AbstractDynamicsModel, ModelBlockCopyMixin, VarGroups

logger = logging.getLogger(__name__)

# TODO benchmark prop speed with and without njit decorator; may not be adding
#   anything and just making code more complex


class Propagator(ModelBlockCopyMixin):
    """
    Create a propagator object.

    This class primarily wraps the :func:`scipy.integrate.solve_ivp` method

    Args:
        model (AbstractDynamicsModel): defines the dynamics model in which the
            propagation occurs
        method (Optional, str): the integration method to use; passed as the
            ``method`` argument to :func:`~scipy.integrate.solve_ivp`
        dense (Optional, bool): whether or not to output dense propagation data;
            passed as the ``dense_output`` argument to
            :func:`~scipy.integrate.solve_ivp`
        atol (Optional, float): absolute tolerance for the integrator; see
            :func:`~scipy.integrate.solve_ivp` for more details
        rtol (Optional, float): relative tolerance for the integrator; see
            :func:`~scipy.integrate.solve_ivp` for more details

    Attributes:
        method (AbstractDynamicsModel): defines the dynamics model
        model (str): the integration method
        dense (bool): whether or not to output dense propagation data
        events ([AbstractEvent]): a list of integration events
    """

    def __init__(self, model, method="DOP853", dense=True, atol=1e-12, rtol=1e-10):
        if not isinstance(model, AbstractDynamicsModel):
            raise ValueError("model must be derived from AbstractDynamicsModel")

        self.method = method
        self.atol = atol
        self.rtol = rtol
        self.model = model
        self.dense = dense
        self.events = []

    def propagate(self, y0, tspan, *, params=None, varGroups=VarGroups.STATE, **kwargs):
        """
        Perform a propagation

        Args:
            y0 (numpy.ndarray<float>): initial state vector
            tspan ([float]): a 2-element vector defining the start and end times
                for the propagation
            params (Optional, [float]): parameters that are passed to the dynamics
                model
            varGroups (Optional, [VarGroups]): the variable groups that are included
                 in ``y0``.
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
            RuntimeError: if the size of ``y0`` is inconsistent with ``varGroups``
                as checked via :func:`AbstractDynamicsModel.stateSize`
            RuntimeError: if any of the objects in :attr:`events` are not derived
                from the :class:`AbstractEvent` base class
        """
        # Checks
        if not self.model.validForPropagation(varGroups):
            raise RuntimeError(f"VarGroups = {varGroups} is not valid for propagation")

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

        # Ensure state is an array with the right number of elements
        y0 = np.array(y0, ndmin=1)
        if not y0.size == self.model.stateSize(varGroups):
            # Likely scenario: user wants default ICs for other variable groups
            if y0.size == self.model.stateSize(VarGroups.STATE):
                y0 = self.model.appendICs(
                    y0, [v for v in varGroups if not v == VarGroups.STATE]
                )
            else:
                raise RuntimeError(
                    f"y0 size is {y0.size}, which is not the STATE size "
                    f"({self.model.stateSize(VarGroups.STATE)}); don't know how to respond."
                    " Please pass in a STATE-sized vector or a vector with all "
                    "initial conditions defined for the specified VarGroups"
                )

        # make varGroups an array and then cast to tuple; need simple type for
        #   JIT compilation support
        varGroups = tuple(sorted(np.array(varGroups, ndmin=1)))

        # TODO also cast params to tuple?
        kwargs_in["args"] = (varGroups, params)

        # Gather event functions and assign attributes
        for event in self.events:
            if not isinstance(event, AbstractEvent):
                raise RuntimeError(f"Event is not derived from AbstractEvent:\n{event}")
            event.assignEvalAttr()

        eventFcns = [event.eval for event in self.events]
        eventFcns.extend(kwargs_in.get("events", []))
        kwargs_in["events"] = eventFcns

        # Run propagation
        sol = solve_ivp(self.model.diffEqs, tspan, y0, **kwargs_in)

        # Append our own metadata
        sol.model = self.model
        sol.params = copy(params)
        sol.varGroups = varGroups

        return sol


class AbstractEvent(ABC):
    """
    Defines an abstract event object

    Args:
        terminal (Optional, bool, int): defines how the event interacts with the
            proapgation. If ``True``, the propagation will stop at the first occurrence
            of this event. If an :class:`int` is passed, the propagation will
            end after the specified number of occurrences.
        direction (Optional, float): defines event direction. If less than zero,
            the event is only triggered when ``eval`` moves from positive to
            negative values. A positive value triggers in the opposite direction,
            and ``0`` will trigger the event in either direction.
    """

    def __init__(self, terminal=False, direction=0.0):
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

        self.terminal = terminal
        self.direction = direction

    def assignEvalAttr(self):
        """
        Must be called before starting propagation so that the
        :attr:`terminal` and :attr:`direction` values take effect
        """
        self.eval.__func__.terminal = self.terminal
        self.eval.__func__.direction = self.direction

    @abstractmethod
    def eval(self, t, y, varGroups, params):
        """
        Event evaluation method

        Args:
            t (float): time value
            y (numpy.ndarray<float>): variable vector
            varGroups (tuple<VarGroups>): describes the variable groups in ``y``
            params ([float]): extra parameters passed from the integrator

        Returns:
            float: the event value. The event occurs when this value is zero
        """
        pass


class ApseEvent(AbstractEvent):
    """
    Occurs at an apsis relative to a body in the model

    Args:
        model (AbstractDynamicsModel): a dynamics model
        bodyIx (int): index of the body
        terminal (Optional, bool, int): defines how the event interacts with the
            proapgation. If ``True``, the propagation will stop at the first occurrence
            of this event. If an :class:`int` is passed, the propagation will
            end after the specified number of occurrences.
        direction (Optional, float): defines event direction. A positive value
            corresponds to periapsis, a negative value corresponds to an
            apoapsis, and a zero value finds all apses.
    """

    def __init__(self, model, bodyIx, terminal=False, direction=0.0):
        if not isinstance(model, AbstractDynamicsModel):
            raise ValueError("model must be derived from AbstractDynamicsModel")

        super().__init__(terminal, direction)
        self._model = model
        self._ix = bodyIx

    def eval(self, t, y, varGroups, params):
        return self._eval(t, y, params, self._model, self._ix)

    @staticmethod
    # @njit  # would need to pass bodyState in as arg (TODO?)
    def _eval(t, y, params, model, ix):
        # apse occurs where dot product between primary-relative position and
        # velocity is zero
        # TODO this assumes the state vector is the Cartesian position and velocity
        #   vectors
        relState = y[:6] - model.bodyState(ix, t, params)
        return (
            relState[0] * relState[3]
            + relState[1] * relState[4]
            + relState[2] * relState[5]
        )


class BodyDistanceEvent(AbstractEvent):
    """
    Occurs when the position distance relative to a body equals a specified value

    Args:
        model (AbstractDynamicsModel): the model that includes the body
        ix (int): index of the body in the model
        dist (float): distance value
        terminal (Optional, bool, int): defines how the event interacts with the
            proapgation. If ``True``, the propagation will stop at the first occurrence
            of this event. If an :class:`int` is passed, the propagation will
            end after the specified number of occurrences.
        direction (Optional, float): defines event direction. If less than zero,
            the event is only triggered when ``eval`` moves from positive to
            negative values. A positive value triggers in the opposite direction,
            and ``0`` will trigger the event in either direction.
    """

    def __init__(self, model, ix, dist, terminal=False, direction=0.0):
        if not isinstance(model, AbstractDynamicsModel):
            raise ValueError("model must be derived from AbstractDynamicsModel")

        super().__init__(terminal, direction)
        self._model = model
        self._ix = ix
        self._dist = dist * dist  # evaluation uses squared value

    def eval(self, t, y, varGroups, params):
        # TODO assumes Cartesian position vector is the first piece of state vector
        relPos = y[:3] - self._model.bodyState(self._ix, t, params)[:3]
        return sum([x * x for x in relPos]) - self._dist


class DistanceEvent(AbstractEvent):
    """
    Occurs when the L2 norm difference between two vectors equals the specified distance

    Args:
        dist (float): the distance
        vec (numpy.ndarray<float>): the goal vector
        ixs (numpy.ndarray<int>): the indices of the variables to compare to ``vec``
        terminal (Optional, bool, int): defines how the event interacts with the
            proapgation. If ``True``, the propagation will stop at the first occurrence
            of this event. If an :class:`int` is passed, the propagation will
            end after the specified number of occurrences.
        direction (Optional, float): defines event direction. If less than zero,
            the event is only triggered when ``eval`` moves from positive to
            negative values. A positive value triggers in the opposite direction,
            and ``0`` will trigger the event in either direction.
    """

    def __init__(self, dist, vec, ixs=[0, 1, 2], terminal=False, direction=0.0):
        vec, ixs = np.array(vec, ndmin=1), np.array(ixs, ndmin=1)
        if not vec.shape == ixs.shape:
            raise ValueError("vec and ixs must have the same shape")

        super().__init__(terminal, direction)
        self._vec = vec
        self._ixs = ixs
        self._dist = dist * dist  # evaluation uses squared value

    def eval(self, t, y, varGroups, params):
        dist = self._vec - y[self._ixs]
        return sum([x * x for x in dist]) - self._dist


class VariableValueEvent(AbstractEvent):
    """
    Occurs where a variable equals a specified value

    Args:
        varIx (int): index of the variable within the array
        varValue (float): the value of the variable
        terminal (Optional, bool, int): defines how the event interacts with the
            proapgation. If ``True``, the propagation will stop at the first occurrence
            of this event. If an :class:`int` is passed, the propagation will
            end after the specified number of occurrences.
        direction (Optional, float): defines event direction. If less than zero,
            the event is only triggered when ``eval`` moves from positive to
            negative values. A positive value triggers in the opposite direction,
            and ``0`` will trigger the event in either direction.

    """

    def __init__(self, varIx, varValue, terminal=False, direction=0.0):
        super().__init__(terminal, direction)
        self._ix = varIx
        self._val = varValue

    def eval(self, t, y, varGroups, params):
        return self._eval(y, self._ix, self._val)

    @staticmethod
    @njit
    def _eval(y, ix, val):
        return val - y[ix]
