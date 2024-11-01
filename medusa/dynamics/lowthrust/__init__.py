"""
Low-Thrust Dynamics
===================

Most spacecraft fly with some sort of propulsion system to accomplish midcourse
adjustments to the trajectory. While "low" is a subjective measure, a low-thrust
propulsion system is usually only capable of generating a few Newtons of thrust.
However, the more dynamically interesting feature of low-thrust propulsion is that
it acts over long time scales. High-thrust systems typically "burn" for minutes or
perhaps hours; a low-thrust system may continue thrusting for days, weeks, or 
even years. As such, the thrusting cannot be modeled as an instantaneous change
in the spacecraft's velocity. Instead, it is more useful to model the low-thrust
force as a time-varying part of the dynamics.

Control Laws
------------

Within this module, low-thrust is described by a **control law** object. The primary
output of this object is an acceleration vector that can be added to other 
accelerations acting on a spacecraft (e.g., gravity).

Recall from the :doc:`dynamics` documentation that the **variable vector** is 
defined as

.. math::
   \\vec{w} = \\begin{Bmatrix}
     \\vec{q} & 
     \\mathbf{\\Phi} &
     \\frac{\\partial \\vec{q}}{\\partial T} &
     \\frac{\\partial \\vec{q}}{\\partial \\vec{p}}
   \\end{Bmatrix},

where :math:`\\vec{q}` is the **state vector**, :math:`\\mathbf{\\Phi}` is the
state transition matrix, and the final two terms are the partial derivatives of
the state vector with respect to the epoch, :math:`T`, and the parameters,
:math:`\\vec{p}`. The :class:`~medusa.dynamics.AbstractDynamicsModel` computes 
the derivative of :math:`\\vec{w}`. Accordingly, any dynamical model incorporating 
a low-thrust control law will need to know how the control law affects each of 
these quantities.


Control States and Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The control law may define control states,
:math:`\\vec{u} = \\vec{u}(\\tau, T, \\vec{p})`, and/or control parameters, 
:math:`\\vec{p}_u`. These quantities are incorporated into the full state and 
parameter vectors,

.. math::
   \\vec{q} &= \\begin{Bmatrix} \\vec{q}_c & \\vec{u} \\end{Bmatrix}, \\\\
   \\vec{p} &= \\begin{Bmatrix} \\vec{p}_c & \\vec{p}_u \\end{Bmatrix},

where :math:`\\vec{q}_c` is the **core state** and includes, at a minimum,
the Cartesian position and velocity coordinates, as in

.. math::
    \\vec{q}_c(\\tau, T, \\vec{p}) =
    \\begin{Bmatrix} x & y & z & \\dot{x} & \\dot{y} & \\dot{z} & \\ldots \\end{Bmatrix}^T.

The core state may also include other variables, but it is assumed that they occur
*after* the Cartesian coordinates.

.. note::
   This restriction that the dynamical model parameterize the state via Cartesian
   coordinates is not strictly necessary but it simplifies the calculations and
   allows terms to be defined independently of one another.

The :math:`\\vec{p}_c` vector lists the **core parameters**, i.e., the model 
parameters that exist independently of the control law. The control states and
control parameters are provided by the control law:

.. list-table::
   :header-rows: 1

   * - Term
     - Method
   * - :math:`\\vec{u}(\\tau = 0)`
     - :attr:`ControlLaw.stateICs`
   * - :math:`\\vec{p}_u`
     - :attr:`ControlLaw.params`

.. important::
   The parameter values defined by the control law are only the *default*
   values; during a numerical propagation, the parameter values must be passed to 
   the propagator explicitly, e.g.,

   .. code-block:: python

      controlLaw = MyControl(...)
      model = MyModel(..., controlLaw)

      sol = propagator.propagate(
          initialCondition,
          tspan,
          params = model.params,   # or provide custom values
      )

   Similarly, many of the :class:`ControlLaw` and :class:`ControlTerm` functions
   rely on parameter values passed to the function and do *not* automatically use
   the default values.


Derivatives
^^^^^^^^^^^

To provide the differential equations for a dynamical model, i.e., 
:math:`\\dot{\\vec{w}}`, a number of derivatives must be supplied by the control
law.

.. note::
   The calculus that follows is not implemented in the control law objects 
   themselves. Rather, these relationships should be coded into the
   :func:`~AbstractDynamicsModel.diffEqs` method with relevant inputs provided
   by the control law.

State Derivative
''''''''''''''''

The control law affects the state derivative by adding an acceleration to the
system as well as its own control state derivatives,

.. math::
   \\dot{\\vec{q}} = \\begin{Bmatrix}
       \\vec{v} \\\\
       \\vec{a}_c + \\vec{a}_u \\\\
       \\vdots \\\\
       \\dot{\\vec{u}}
   \\end{Bmatrix}

where :math:`\\vec{v}` is the velocity vector, :math:`\\vec{a}_c` is the acceleration
due to gravity and other forces, :math:`\\vec{a}_u` is the acceleration delivered
by the low-thrust force, and :math:`\\dot{\\vec{u}}` is the derivative of the 
control state vector. The :math:`\\vec{v}` and :math:`\\vec{a}_c` terms are computed
by the dynamical model, but the control law supplies the other two terms:

.. list-table::
   :header-rows: 1

   * - Term
     - Method
   * - :math:`\\vec{a}_u`
     - :func:`ControlLaw.accelVec`
   * - :math:`\\dot{\\vec{u}}`
     - :func:`ControlLaw.stateDiffEqs`

State Partials
''''''''''''''

Recall from the :doc:`dynamics` documentation that the state transition matrix is 
described by the differential equation,

.. math::
   \\dot{\\mathbf{\\Phi}}(\\tau_2, \\tau_1) = \\mathbf{A}(\\tau_2) \\mathbf{\\Phi}(\\tau_2, \\tau_1)

where :math:`\\mathbf{A}` is the linearization of the dynamics,

.. math::
   \\mathbf{A} = \\frac{ \\partial \\dot{\\vec{q}} }{ \\partial \\vec{q} }.

Separating the core and control state variables, the :math:`\\mathbf{A}` matrix 
can be written as four submatrices,

.. math::
   \\mathbf{A} = \\begin{bmatrix}
     \\partial \\dot{\\vec{q}}_c / \\partial \\vec{q}_c & \\partial \\dot{\\vec{q}}_c / \\partial \\vec{u} \\\\
     \\partial \\dot{\\vec{u}} / \\partial \\vec{q}_c & \\partial \\dot{\\vec{u}} / \\partial \\vec{u}
   \\end{bmatrix}

The terms in the top row of this block matrix can be further decomposed, leveraging
the definition of :math:`\\dot{\\vec{q}}` above,

.. math::
   \\partial \\dot{\\vec{q}}_c / \\partial \\vec{q}_c &=
     (\\partial \\dot{\\vec{q}}_c / \\partial \\vec{a}_u)
     (\\partial \\vec{a}_u / \\partial \\vec{q}_c) \\\\
   \\partial \\dot{\\vec{q}}_c / \\partial \\vec{u} &=
     (\\partial \\dot{\\vec{q}}_c / \\partial \\vec{a}_u)
     (\\partial \\vec{a}_u / \\partial \\vec{u}) \\\\

where

.. math::
   \\partial \\dot{\\vec{q}}_c / \\partial \\vec{a}_u = \\begin{bmatrix}
     \\mathbf{0}_{3 \\times 3} \\\\
     \\mathbf{I}_3 \\\\
     \\mathbf{0}_{ n \\times 3}
   \\end{bmatrix}.

This expansion yields four terms that must be supplied by the control law:

.. list-table::
   :header-rows: 1

   * - Term
     - Method
   * - :math:`\\partial \\vec{a}_u / \\partial \\vec{q}_c`
     - :func:`ControlLaw.partials_accel_wrt_coreState`
   * - :math:`\\partial \\vec{a}_u / \\partial \\vec{u}`
     - :func:`ControlLaw.partials_accel_wrt_ctrlState`
   * - :math:`\\partial \\dot{\\vec{u}} / \\partial \\vec{q}_c`
     - :func:`ControlLaw.partials_ctrlStateDEQs_wrt_coreState`
   * - :math:`\\partial \\dot{\\vec{u}} / \\partial \\vec{u}`
     - :func:`ControlLaw.partials_ctrlStateDEQs_wrt_ctrlState`

Epoch Partials
''''''''''''''

Similar to the state partials, the epoch partials are modified by the control law.
The dynamical model differential equations quantify the derivative,

.. math::
   \\frac{\\partial \\dot{\\vec{q}}}{\\partial T} = \\begin{Bmatrix}
     \\partial \\dot{\\vec{q}}_c / \\partial T +
       (\\partial \\dot{\\vec{q}}_c / \\partial \\vec{a}_u)
       (\\partial \\vec{a}_u / \\partial T) \\\\
     \\partial \\dot{\\vec{u}} / \\partial T
   \\end{Bmatrix}

The partial derivative of :math:`\\dot{\\vec{q}}_c` with respect to the epoch is
provided by the dynamical model.
The partial derivatives of the acceleration vector and the control state
differential equations with respect to the epoch are provided by the control law:

.. list-table::
   :header-rows: 1

   * - Term
     - Method
   * - :math:`\\partial \\vec{a}_u / \\partial T`
     - :func:`ControlLaw.partials_accel_wrt_epoch`
   * - :math:`\\partial \\dot{\\vec{u}} / \\partial T`
     - :func:`ControlLaw.partials_ctrlStateDEQs_wrt_epoch`

Parameter Partials
''''''''''''''''''

The parameter partials are computed in the same way as the epoch partials,

.. math::
   \\frac{\\partial \\dot{\\vec{q}}}{\\partial \\vec{p}} = \\begin{Bmatrix}
     \\partial \\dot{\\vec{q}}_c / \\partial \\vec{p} +
       (\\partial \\dot{\\vec{q}}_c / \\partial \\vec{a}_u)
       (\\partial \\vec{a}_u / \\partial \\vec{p}) \\\\
     \\partial \\dot{\\vec{u}} / \\partial \\vec{p}
   \\end{Bmatrix}


The partial derivative of :math:`\\dot{\\vec{q}}_c` with respect to the full 
parameter is provided by the dynamical model.

.. note::
   Because the "ballistic" (no thrust applied) dynamics described by 
   :math:`\\dot{\\vec{q}}_c` are independent of the control parameters,
   :math:`\\vec{p}_u`, the partial derivatives defined in the dynamics model
   will likely need to incorporate zeros for 
   :math:`\\partial \\dot{\\vec{q}}_c / \\partial \\vec{p}`.

The partial derivatives of the acceleration vector and the control state
differential equations with respect to the parameters are provided by the control 
law:

.. list-table::
   :header-rows: 1

   * - Term
     - Method
   * - :math:`\\partial \\vec{a}_u / \\partial \\vec{p}`
     - :func:`ControlLaw.partials_accel_wrt_params`
   * - :math:`\\partial \\dot{\\vec{u}} / \\partial \\vec{p}`
     - :func:`ControlLaw.partials_ctrlStateDEQs_wrt_params`


Separable Control Parameterizations
-----------------------------------

In many contexts, the low-thrust control is easily separable into independent terms
like thrust force and vector orientation. The :class:`SeparableControlLaw` provides
this type of parameterization via an arbitrary number of :class:`ControlTerm`
objects.

.. autosummary::
   SeparableControlLaw
   ControlTerm

User scripts can use this architecture to flexibly define a control 
parameterization. Each ``ControlTerm`` object defines its own parameters
and control states, as well as the relevant partial derivatives. The 
``SeparableControlLaw`` concatenates all of the control states and parameters
together, as well as a subset of the partial derivatives. A full combination of 
the partial derivatives requires a definition of the acceleration vector, however.
The :class:`SeparableControlLaw` is, thus, an abstract class; derived classes must
define the :func:`~ControlLaw.accelVec` function and all of the functions that
compute partial derivatives of that acceleration vector.

An even more specific parameterization with three terms -- one for thrust force,
one for spacecraft mass, and another for thrust orientation -- is available via
the :class:`ForceMassOrientLaw` with some convenient terms pre-defined.

.. autosummary::
   ForceMassOrientLaw
   ConstThrustTerm
   ConstMassTerm
   ConstOrientTerm

Because the acceleration vector is fully defined, this implementation defines
the full set of partial derivatives. Custom parameterizations of the thrust
magnitude, mass, or thrust vector orientation can be "plugged in" to alter the
control law behavior.

Dynamical Models
----------------

As discussed above, a :class:`ControlLaw` is meant to be an input to a dynamical
model. For example, a CR3BP model (see :doc:`dynamics.crtbp`) can be augmented
with any control law to yield a CR3BP+LT model.

.. toctree::
   :maxdepth: 1

   dynamics.lowthrust.crtbp

Reference
===========

.. autoclass:: ControlLaw
   :members:
   :show-inheritance:

.. autoclass:: SeparableControlLaw
   :members:
   :show-inheritance:

.. autoclass:: ControlTerm
   :members:
   :show-inheritance:

.. autoclass:: ConstThrustTerm
   :members:
   :show-inheritance:

.. autoclass:: ConstMassTerm
   :members:
   :show-inheritance:

.. autoclass:: ConstOrientTerm
   :members:
   :show-inheritance:

.. autoclass:: ForceMassOrientLaw
   :members:
   :show-inheritance:
"""

__all__ = [
    # base module
    "ControlTerm",
    "ConstThrustTerm",
    "ConstMassTerm",
    "ConstOrientTerm",
    "ControlLaw",
    "SeparableControlLaw",
    "ForceMassOrientLaw",
    # sub modules
    "crtbp",
]

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from medusa import util
from medusa.corrections import Variable
from medusa.dynamics import VarGroup
from medusa.typing import FloatArray, override

# ------------------------------------------------------------------------------
# Abstract Objects
# ------------------------------------------------------------------------------


class ControlLaw(ABC):
    """
    Interface definition for a low-thrust control law.
    """

    def __init__(self) -> None:
        self._coreStateSize: Union[None, int] = None
        self._paramIx0: Union[None, int] = None

    @property
    @abstractmethod
    def epochIndependent(self) -> bool:
        """Whether or not the control law is epoch-independent"""
        pass

    @property
    @abstractmethod
    def numStates(self) -> int:
        """The number of control state variables defined by the control law"""
        pass

    @property
    @abstractmethod
    def stateNames(self) -> list[str]:
        """The names of the control state variables defined by the control law"""
        pass

    @property
    def stateICs(self) -> NDArray[np.double]:
        """
        Initial conditions for the control state variables defined by the control law.

        This implementation returns zeros for the control states; derived classes
        can override this behavior.
        """
        if self.numStates == 0:
            return np.asarray([])
        else:
            return np.zeros((self.numStates,))

    @property
    @abstractmethod
    def params(self) -> FloatArray:
        """Default values for the control parameters"""
        pass

    def register(self, nCore: int, ix0: int) -> None:
        """
        Register the control law within the context of the full dynamics model

        Args:
            nCore (int): the number of core states, i.e., the number of state
                variables excluding the control states
            ix0 (int): the index of the first control parameter within the full
                set of parameters.
        """
        self._coreStateSize = nCore
        self._paramIx0 = ix0

    @abstractmethod
    def accelVec(
        self,
        t: float,
        w: NDArray[np.double],
        varGroups: tuple[VarGroup, ...],
        params: Union[tuple[float, ...], None],
    ) -> NDArray[np.double]:
        """
        Compute the acceleration vector delivered by this control law.

        The input arguments are consistent with those passed to the
        :func:`medusa.dynamics.AbstractDynamicsModel.diffEqs` function.

        Returns:
            numpy.ndarray: a 3x1 array that gives the Cartesian acceleration
            vector.
        """
        pass

    @abstractmethod
    def stateDiffEqs(
        self,
        t: float,
        w: NDArray[np.double],
        varGroups: tuple[VarGroup, ...],
        params: Union[tuple[float, ...], None],
    ) -> NDArray[np.double]:
        """
        Defines the differential equations that govern the state variables this
        control law defines, i.e., derivatives of the state variables with respect
        to integration time.

        The input arguments are consistent with those passed to the
        :func:`medusa.dynamics.AbstractDynamicsModel.diffEqs` function.

        Returns:
            the time derivatives of the state variables. If this
            term doesn't define any state variables, an empty array is returned.
        """
        pass

    @abstractmethod
    def partials_accel_wrt_coreState(
        self,
        t: float,
        w: NDArray[np.double],
        varGroups: tuple[VarGroup, ...],
        params: Union[tuple[float, ...], None],
    ) -> NDArray[np.double]:
        """
        The partial derivatives of :func:`accelVec` with respect to the "core state,"
        i.e., the state variables that exist independent of the control
        parameterization.

        The input arguments are consistent with those passed to the
        :func:`medusa.dynamics.AbstractDynamicsModel.diffEqs` function.

        Returns:
            the partial derivatives; the rows represent the elements of the
            acceleration vector and the columns represent the core state variables.
        """
        pass

    @abstractmethod
    def partials_accel_wrt_ctrlState(
        self,
        t: float,
        w: NDArray[np.double],
        varGroups: tuple[VarGroup, ...],
        params: Union[tuple[float, ...], None],
    ) -> NDArray[np.double]:
        """
        The partial derivatives of :func:`accelVec` with respect to the control
        states defined by the control law.

        The input arguments are consistent with those passed to the
        :func:`medusa.dynamics.AbstractDynamicsModel.diffEqs` function.

        Returns:
            the partial derivatives; the rows represent the elements of the
            acceleration vector and the columns represent the control state variables.
        """
        pass

    @abstractmethod
    def partials_accel_wrt_epoch(
        self,
        t: float,
        w: NDArray[np.double],
        varGroups: tuple[VarGroup, ...],
        params: Union[tuple[float, ...], None],
    ) -> NDArray[np.double]:
        """
        The partial derivatives of :func:`accelVec` with respect to the epoch.

        The input arguments are consistent with those passed to the
        :func:`medusa.dynamics.AbstractDynamicsModel.diffEqs` function.

        Returns:
            the partial derivatives; the rows represent the elements
            of the acceleration vector and the column represents the epoch.
        """
        pass

    @abstractmethod
    def partials_accel_wrt_params(
        self,
        t: float,
        w: NDArray[np.double],
        varGroups: tuple[VarGroup, ...],
        params: Union[tuple[float, ...], None],
    ) -> NDArray[np.double]:
        """
        The partial derivatives of :func:`accelVec` with respect to the full set
        of parameters.

        The input arguments are consistent with those passed to the
        :func:`medusa.dynamics.AbstractDynamicsModel.diffEqs` function.

        .. note::
           Control parameterizations are generally not a function of parameters
           defined by the "ballistic" model (the model without low thrust), so
           those partials are often trivially zero.

        Returns:
            the partial derivatives; the rows represent the elements
            of the acceleration vector and the columns represent the parameters.
        """
        pass

    @abstractmethod
    def partials_ctrlStateDEQs_wrt_coreState(
        self,
        t: float,
        w: NDArray[np.double],
        varGroups: tuple[VarGroup, ...],
        params: Union[tuple[float, ...], None],
    ) -> NDArray[np.double]:
        """
        The partial derivatives of :func:`stateDiffEqs` with respect to the "core
        states," i.e., the state variables that exist independently of the
        control parameterization.

        The input arguments are consistent with those passed to the
        :func:`medusa.dynamics.AbstractDynamicsModel.diffEqs` function.

        Returns:
            the partial derivatives; the rows represent the differential equations
            and the columns represent the core state variables.
        """
        pass

    @abstractmethod
    def partials_ctrlStateDEQs_wrt_ctrlState(
        self,
        t: float,
        w: NDArray[np.double],
        varGroups: tuple[VarGroup, ...],
        params: Union[tuple[float, ...], None],
    ) -> NDArray[np.double]:
        """
        The partial derivatives of :func:`stateDiffEqs` with respect to the state
        variables defined by the control law.

        The input arguments are consistent with those passed to the
        :func:`medusa.dynamics.AbstractDynamicsModel.diffEqs` function.

        Returns:
            the partial derivatives; the rows represent the differential equations
            and the columns represent the control state variables.
        """
        pass

    @abstractmethod
    def partials_ctrlStateDEQs_wrt_epoch(
        self,
        t: float,
        w: NDArray[np.double],
        varGroups: tuple[VarGroup, ...],
        params: Union[tuple[float, ...], None],
    ) -> NDArray[np.double]:
        """
        The partial derivatives of :func:`stateDiffEqs` with respect to the epoch.

        The input arguments are consistent with those passed to the
        :func:`medusa.dynamics.AbstractDynamicsModel.diffEqs` function.

        Returns:
            the partial derivatives; the rows represent the
            differential equations and the column represents the epoch.
        """
        pass

    @abstractmethod
    def partials_ctrlStateDEQs_wrt_params(
        self,
        t: float,
        w: NDArray[np.double],
        varGroups: tuple[VarGroup, ...],
        params: Union[tuple[float, ...], None],
    ) -> NDArray[np.double]:
        """
        The partial derivatives of :func:`stateDiffEqs` with respect to the
        full set of parameters.

        The input arguments are consistent with those passed to the
        :func:`medusa.dynamics.AbstractDynamicsModel.diffEqs` function.

        .. note::
           Control parameterizations are generally not a function of parameters
           defined by the "ballistic" model (the model without low thrust), so
           those partials are often trivially zero.

        Returns:
            the partial derivatives; the rows represent the
            differential equations and the columns represent the parameters.
        """
        pass


class ControlTerm(ABC):
    """
    Represents a term in the control equations

    .. note:: This is an abstract class and cannot be instantiated.
    """

    def __init__(self) -> None:
        self._coreStateSize: Union[None, int] = None
        self._paramIx0: Union[None, int] = None

    @property
    def epochIndependent(self) -> bool:
        """
        Whether or not this term is independent of the epoch.

        Returns True; override to customize the behavior.
        """
        return True

    @property
    def params(self) -> Sequence[float]:
        """
        The default parameter values

        Returns an empty list; override to customize the behavior.
        """
        return []

    @property
    def numStates(self) -> int:
        """
        The number of control state variables defined by this term.

        Returns zero; override to customize the behavior.
        """
        return 0

    @property
    def stateICs(self) -> NDArray[np.double]:
        """
        The initial conditions for the control state variables this term defines.

        Returns zeros for all control states; override to customize the behavior.
        """
        if self.numStates == 0:
            return np.asarray([])
        else:
            return np.zeros((self.numStates,))

    @property
    def stateNames(self) -> list[str]:
        """
        The names of the control state variables defined by this term.

        Uses the format "{CLASS} {ix}" to name the variables where {CLASS} is
        the name of derived class and {ix} is the index of the variable within
        the control state variables defined by this term. Override to customize
        the behavior.
        """
        me = self.__class__.__name__
        return [f"{me} {ix}" for ix in range(self.numStates)]

    def register(self, nCore: int, ix0: int) -> None:
        """
        Register the control law within the context of the full dynamics model

        Args:
            nCore: the number of core states, i.e., the number of state
                variables excluding the control states
            ix0: the index of the first control parameter within the full
                set of parameters.
        """
        self._coreStateSize = nCore
        self._paramIx0 = ix0

    # Similar to AbstractDynamicsModel.diffEqs, this function and those that
    # follow may be called millions of times, so types are restricted for
    # computational speed
    def stateDiffEqs(
        self,
        t: float,
        w: NDArray[np.double],
        varGroups: tuple[VarGroup, ...],
        params: Union[tuple[float, ...], None],
    ) -> NDArray[np.double]:
        """
        Defines the differential equations that govern the state variables this
        term defines, i.e., derivatives of the state variables with respect to
        integration time.

        The input arguments are consistent with those passed to the
        :func:`medusa.dynamics.AbstractDynamicsModel.diffEqs` function.

        .. note:: This method is implemented to return zeros for all state variables
           by default. Override it to define custom behavior.

        Returns:
            the time derivatives of the state variables. If this
            term doesn't define any state variables, an empty array is returned.
        """
        if self.numStates == 0:
            return np.asarray([])
        else:
            return np.zeros((self.numStates,))

    @abstractmethod
    def evalTerm(
        self,
        t: float,
        w: NDArray[np.double],
        varGroups: tuple[VarGroup, ...],
        params: Union[tuple[float, ...], None],
    ) -> Union[float, NDArray[np.double]]:
        """
        Evaluate the term.

        The input arguments are consistent with those passed to the
        :func:`medusa.dynamics.AbstractDynamicsModel.diffEqs` function.

        Returns:
            the evaluated term
        """
        pass

    @abstractmethod
    def partials_term_wrt_coreState(
        self,
        t: float,
        w: NDArray[np.double],
        varGroups: tuple[VarGroup, ...],
        params: Union[tuple[float, ...], None],
    ) -> NDArray[np.double]:
        """
        Compute the partial derivatives of :func:`evalTerm` with respect to the
        "core state", i.e., the state variables that exist independently of the
        control parametrization.

        The input arguments are consistent with those passed to the
        :func:`medusa.dynamics.AbstractDynamicsModel.diffEqs` function.

        Returns:
            the partial derivatives where the rows represent the
            elements in :func:`evalTerm` and the columns represent the core states.
        """
        pass

    @abstractmethod
    def partials_term_wrt_ctrlState(
        self,
        t: float,
        w: NDArray[np.double],
        varGroups: tuple[VarGroup, ...],
        params: Union[tuple[float, ...], None],
    ) -> NDArray[np.double]:
        """
        Compute the partial derivatives of :func:`evalTerm` with respect to the
        control state variables that are defined by *this term*.

        The input arguments are consistent with those passed to the
        :func:`medusa.dynamics.AbstractDynamicsModel.diffEqs` function.

        Returns:
            the partial derivatives
        """
        pass

    @abstractmethod
    def partials_term_wrt_epoch(
        self,
        t: float,
        w: NDArray[np.double],
        varGroups: tuple[VarGroup, ...],
        params: Union[tuple[float, ...], None],
    ) -> NDArray[np.double]:
        """
        Compute the partial derivatives of :func:`evalTerm` with respect to the epoch.

        The input arguments are consistent with those passed to the
        :func:`medusa.dynamics.AbstractDynamicsModel.diffEqs` function.

        Returns:
            the partial derivatives where the rows represent the
            elements in :func:`evalTerm` and the column represents the epoch.
        """
        pass

    @abstractmethod
    def partials_term_wrt_params(
        self,
        t: float,
        w: NDArray[np.double],
        varGroups: tuple[VarGroup, ...],
        params: Union[tuple[float, ...], None],
    ) -> NDArray[np.double]:
        """
        Compute the partial derivatives of :func:`evalTerm` with respect to the
        parameters *this term* defines.

        The input arguments are consistent with those passed to the
        :func:`medusa.dynamics.AbstractDynamicsModel.diffEqs` function.

        Returns:
            the partial derivatives where the rows represent the
            elements in :func:`evalTerm` and the columns represent the parameters.
        """
        pass

    def partials_coreStateDEQs_wrt_ctrlState(
        self,
        t: float,
        w: NDArray[np.double],
        varGroups: tuple[VarGroup, ...],
        params: Union[tuple[float, ...], None],
    ) -> NDArray[np.double]:
        """
        Compute the partial derivatives of the core state differential equations
        (defined in :func:`~medusa.dynamics.AbstractDynamicsModel.diffEqs`) with
        respect to the control state variables that are defined by this term.

        The input arguments are consistent with those passed to the
        :func:`medusa.dynamics.AbstractDynamicsModel.diffEqs` function.

        Returns:
            the partial derivatives where the rows represent the
            core states and the columns represent the control states defined by
            this term. If this term doesn't define any control states, an
            empty array is returned.
        """
        if self.numStates == 0:
            return np.asarray([])
        else:
            if self._coreStateSize is None:
                raise RuntimeError("core state size has not been set; call register()")

            return np.zeros((self._coreStateSize, self.numStates))

    def partials_ctrlStateDEQs_wrt_coreState(
        self,
        t: float,
        w: NDArray[np.double],
        varGroups: tuple[VarGroup, ...],
        params: Union[tuple[float, ...], None],
    ) -> NDArray[np.double]:
        """
        Compute the partial derivatives of :func:`stateDiffEqs` with respect to
        the "core state," i.e., the state variables that exist independent of the
        control parameterization.

        The input arguments are consistent with those passed to the
        :func:`medusa.dynamics.AbstractDynamicsModel.diffEqs` function.

        Returns:
            the partial derivatives where the rows represent the
            elements in :func:`stateDiffEqs` and the columns represent the core
            states. If this term doesn't define any control states, an empty
            array is returned.
        """
        if self.numStates == 0:
            return np.asarray([])
        else:
            if self._coreStateSize is None:
                raise RuntimeError("core state size has not been set; call register()")

            return np.zeros((self.numStates, self._coreStateSize))

    def partials_ctrlStateDEQs_wrt_ctrlState(
        self,
        t: float,
        w: NDArray[np.double],
        varGroups: tuple[VarGroup, ...],
        params: Union[tuple[float, ...], None],
    ) -> NDArray[np.double]:
        """
        Compute the partial derivatives of :func:`stateDiffEqs` with respect to
        the control state variables defined by this term.

        The input arguments are consistent with those passed to the
        :func:`medusa.dynamics.AbstractDynamicsModel.diffEqs` function.

        Returns:
            the partial derivatives where the rows represent the
            elements in :func:`stateDiffEqs` and the columns represent the core
            states. If this term doesn't define any control states, an empty
            array is returned.
        """
        if self.numStates == 0:
            return np.asarray([])
        else:
            return np.zeros((self.numStates, self.numStates))

    def partials_ctrlStateDEQs_wrt_epoch(
        self,
        t: float,
        w: NDArray[np.double],
        varGroups: tuple[VarGroup, ...],
        params: Union[tuple[float, ...], None],
    ) -> NDArray[np.double]:
        """
        Compute the partial derivatives of :func:`stateDiffEqs` with respect to
        the epoch.

        The input arguments are consistent with those passed to the
        :func:`medusa.dynamics.AbstractDynamicsModel.diffEqs` function.

        Returns:
            the partial derivatives where the rows represent the
            elements in :func:`stateDiffEqs` and the column represents epoch.
            If this term doesn't define any control states, an empty array is
            returned.
        """
        if self.numStates == 0:
            return np.asarray([])
        else:
            return np.zeros((self.numStates,))

    def partials_ctrlStateDEQs_wrt_params(
        self,
        t: float,
        w: NDArray[np.double],
        varGroups: tuple[VarGroup, ...],
        params: Union[tuple[float, ...], None],
    ) -> NDArray[np.double]:
        """
        Compute the partial derivatives of :func:`stateDiffEqs` with respect to
        the parameters defined by this term.

        The input arguments are consistent with those passed to the
        :func:`medusa.dynamics.AbstractDynamicsModel.diffEqs` function.

        Returns:
            the partial derivatives where the rows represent the
            elements in :func:`stateDiffEqs` and the columns represent the
            parameters. If this term doesn't define any control states, an empty
            array is returned.
        """
        if self.numStates == 0 or params is None or len(params) == 0:
            return np.asarray([])
        else:
            return np.zeros((self.numStates, len(params)))


class SeparableControlLaw(ControlLaw):
    """
    A control law implementation with "separable" terms. In this context,
    separable means that each term defines state variables and parameters
    independently of the other terms. As a result, the control state
    differential equations and their partial derivatives can be concatenated
    without additional calculus to relate the terms.

    .. note:: This implementation is abstract; derived objects must define the
       acceleration vector and its associated partial derivatives.

    Args:
        terms: the term(s) to include in the control parameterization.

    All outputs related to parameters or control states
    are concatenated in an order consistent with the order the terms are passed
    to the constructor.
    """

    def __init__(self, *terms: ControlTerm) -> None:
        self.terms = tuple(terms)

    def _concat(self, arrays: Sequence[ArrayLike]) -> NDArray:
        """
        Concatenate numpy arrays. This convenience method skips concatenation of
        empty arrays, avoiding errors.
        """
        out = np.array(arrays[0], copy=True)
        for array in arrays[1:]:
            if np.asarray(array).size > 0:
                out = np.concatenate((out, array))

        return out

    def __repr__(self) -> str:
        out = f"<{self.__class__.__module__}.{self.__class__.__name__}: terms=("
        for term in self.terms:
            out += "\n  {!r},".format(term)
        out += "\n)>"
        return out

    @property
    @override
    def epochIndependent(self) -> bool:
        """
        True if all of the terms are epoch independent, False otherwise.
        """
        return all(term.epochIndependent for term in self.terms)

    @property
    @override
    def numStates(self) -> int:
        """
        The total number of control state variables defined by the terms
        included in this law.
        """
        return sum(term.numStates for term in self.terms)

    @property
    @override
    def stateNames(self) -> list[str]:
        """
        The names of the control state variables for all terms included in
        this law.
        """
        return util.toList(self._concat([term.stateNames for term in self.terms]))

    @property
    @override
    def params(self) -> NDArray[np.double]:
        """
        The full set of parameters from the terms included in this law.
        """
        return self._concat([term.params for term in self.terms])

    @property
    @override
    def stateICs(self) -> NDArray[np.double]:
        """
        Returns:
            Initial conditions for the full set of control state variables
        """
        return self._concat([term.stateICs for term in self.terms])

    @override
    def stateDiffEqs(self, t, w, varGroups, params) -> NDArray[np.double]:
        """
        The input arguments are consistent with those passed to the
        :func:`medusa.dynamics.AbstractDynamicsModel.diffEqs` function.

        Returns:
            Differential equations governing the full set of control state variables
            for all terms included in this law.
        """
        return self._concat(
            [term.stateDiffEqs(t, w, varGroups, params) for term in self.terms]
        )

    @override
    def register(self, nCore, ix0) -> None:
        """
        Register the control law within the context of the dynamics model

        Args:
            nCore: the number of core states, i.e., the number of state
                variables excluding the control states
            ix0: the index of the first control parameter within the full
                set of parameters.
        """
        super().register(nCore, ix0)

        for term in self.terms:
            term.register(nCore, ix0)
            ix0 += len(term.params)

    @override
    def partials_ctrlStateDEQs_wrt_coreState(
        self, t, w, varGroups, params
    ) -> NDArray[np.double]:
        # Because the control terms are independent, we can just concatenate
        #   the partial derivatives of the control state diff eqs.
        return self._concat(
            [
                term.partials_ctrlStateDEQs_wrt_coreState(t, w, varGroups, params)
                for term in self.terms
            ]
        )

    @override
    def partials_ctrlStateDEQs_wrt_ctrlState(
        self, t, w, varGroups, params
    ) -> NDArray[np.double]:
        return self._concat(
            [
                term.partials_ctrlStateDEQs_wrt_ctrlState(t, w, varGroups, params)
                for term in self.terms
            ]
        )

    @override
    def partials_ctrlStateDEQs_wrt_epoch(
        self, t, w, varGroups, params
    ) -> NDArray[np.double]:
        return self._concat(
            [
                term.partials_ctrlStateDEQs_wrt_epoch(t, w, varGroups, params)
                for term in self.terms
            ]
        )

    @override
    def partials_ctrlStateDEQs_wrt_params(
        self, t, w, varGroups, params
    ) -> NDArray[np.double]:
        return self._concat(
            [
                term.partials_ctrlStateDEQs_wrt_params(t, w, varGroups, params)
                for term in self.terms
            ]
        )


# ------------------------------------------------------------------------------
# Concrete Implementations
# ------------------------------------------------------------------------------


class ConstThrustTerm(ControlTerm):
    """
    Defines a constant thrust.

    Args:
        thrust: the thrust force in units consistent with the model
            (i.e., if the model nondimensionalizes values, this value should
            also be nondimensionalized).

    .. list-table:: Term Properties
       :stub-columns: 1
       :widths: 30 70

       * - Parameters
         - 1, the thrust
       * - Control States
         - *None*
       * - Epoch Dependent?
         - False

    """

    def __init__(self, thrust: float) -> None:
        super().__init__()
        self.thrust = thrust

    def __repr__(self) -> str:
        return "<ConstThrustTerm: thrust={!r}>".format(self.thrust)

    @property
    @override
    def params(self) -> list[float]:
        return [self.thrust]

    @override
    def evalTerm(self, t, w, varGroups, params) -> float:
        return params[self._paramIx0]

    @override
    def partials_term_wrt_coreState(
        self, t, w, varGroups, params
    ) -> NDArray[np.double]:
        if self._coreStateSize is None:
            raise RuntimeError("core state size has not been set; call register()")

        return np.zeros((1, self._coreStateSize))

    @override
    def partials_term_wrt_ctrlState(
        self, t, w, varGroups, params
    ) -> NDArray[np.double]:
        return np.asarray([])  # No control states

    @override
    def partials_term_wrt_epoch(self, t, w, varGroups, params) -> NDArray[np.double]:
        return np.array([0], ndmin=2)

    @override
    def partials_term_wrt_params(self, t, w, varGroups, params) -> NDArray[np.double]:
        partials = np.zeros((1, len(params)))
        partials[0, self._paramIx0] = 1
        return partials


class ConstMassTerm(ControlTerm):
    """
    Defines a constant mass.

    Args:
        mass: the mass in units consistent with the model
            (i.e., if the model nondimensionalizes values, this value should
            also be nondimensionalized).

    .. list-table:: Term Properties
       :stub-columns: 1
       :widths: 30 70

       * - Parameters
         - 1, the mass
       * - Control States
         - *None*
       * - Epoch Dependent?
         - False

    """

    def __init__(self, mass: float) -> None:
        super().__init__()
        self.mass = mass

    def __repr__(self) -> str:
        return "<ConstMassTerm: mass={!r}>".format(self.mass)

    @property
    @override
    def params(self) -> list[float]:
        return [self.mass]

    @override
    def evalTerm(self, t, w, varGroups, params) -> float:
        return params[self._paramIx0]

    @override
    def partials_term_wrt_coreState(
        self, t, w, varGroups, params
    ) -> NDArray[np.double]:
        if self._coreStateSize is None:
            raise RuntimeError("core state size has not been set; call register()")

        return np.zeros((1, self._coreStateSize))

    @override
    def partials_term_wrt_ctrlState(
        self, t, w, varGroups, params
    ) -> NDArray[np.double]:
        return np.asarray([])  # No control states

    @override
    def partials_term_wrt_epoch(self, t, w, varGroups, params) -> NDArray[np.double]:
        return np.array([0], ndmin=2)

    @override
    def partials_term_wrt_params(self, t, w, varGroups, params) -> NDArray[np.double]:
        partials = np.zeros((1, len(params)))
        partials[0, self._paramIx0] = 1
        return partials


class ConstOrientTerm(ControlTerm):
    """
    Defines a constant thrust orientation in the working frame. Orientation
    is parameterized via spherical angles alpha and beta.

    Args:
        alpha: the angle between the projection of the thrust vector
            into the xy-plane and the x-axis, measured about the z-axis. Units
            are radians.
        beta: the angle between the thrust vector and the xy-plane. A
            positive value corresponds to a positive z-component. Units are
            radians.

    .. list-table:: Term Properties
       :stub-columns: 1
       :widths: 30 70

       * - Parameters
         - 2, the ``alpha`` and ``beta`` angles
       * - Control States
         - *None*
       * - Epoch Dependent?
         - False

    """

    def __init__(self, alpha: float, beta: float) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def __repr__(self) -> str:
        return "<ConstOrientTerm: alpha={!r}, beta={!r}>".format(self.alpha, self.beta)

    @property
    @override
    def params(self) -> list[float]:
        return [self.alpha, self.beta]

    def _getAngles(self, params: Sequence[float]) -> tuple[float, float]:
        if self._paramIx0 is None:
            raise RuntimeError("paramIx0 has not been set; call register()")

        return params[self._paramIx0], params[self._paramIx0 + 1]

    @override
    def evalTerm(self, t, w, varGroups, params) -> NDArray[np.double]:
        alpha, beta = self._getAngles(params)
        return np.asarray(
            [
                [np.cos(beta) * np.cos(alpha)],
                [np.cos(beta) * np.sin(alpha)],
                [np.sin(beta)],
            ]
        )

    @override
    def partials_term_wrt_coreState(
        self, t, w, varGroups, params
    ) -> NDArray[np.double]:
        if self._coreStateSize is None:
            raise RuntimeError("core state size has not been set; call register()")

        return np.zeros((3, self._coreStateSize))

    @override
    def partials_term_wrt_ctrlState(
        self, t, w, varGroups, params
    ) -> NDArray[np.double]:
        return np.asarray([])  # no control states

    @override
    def partials_term_wrt_epoch(self, t, w, varGroups, params) -> NDArray[np.double]:
        return np.zeros((3, 1))

    @override
    def partials_term_wrt_params(self, t, w, varGroups, params) -> NDArray[np.double]:
        if self._paramIx0 is None:
            raise RuntimeError("paramIx0 has not been set; call register()")

        partials = np.zeros((3, len(params)))
        alpha, beta = self._getAngles(params)
        partials[:, self._paramIx0] = [
            -np.cos(beta) * np.sin(alpha),
            np.cos(beta) * np.cos(alpha),
            0,
        ]
        partials[:, self._paramIx0 + 1] = [
            -np.sin(beta) * np.cos(alpha),
            -np.sin(beta) * np.sin(alpha),
            np.cos(beta),
        ]
        return partials


class ForceMassOrientLaw(SeparableControlLaw):
    """
    A separable control law that accepts three terms: force, mass, and orientation.

    Args:
        force: defines the scalar thrust force
        mass: defines the scalar mass
        orient: defines the unit vector that orients the thrust

    The acceleration is computed as

    .. math::
       \\vec{a} = \\frac{f}{m} \\hat{a}

    where :math:`f` is the thrust force from the ``force`` term, :math:`m` is the
    mass from the ``mass`` term, and :math:`\\hat{a}` is the orientation from the
    ``orient`` term.

    .. note::
       The control terms should define the thrust and mass in units consistent
       with the dynamical model the control law is included in; these are
       usually nondimensional values.

    .. list-table:: Control Law Properties
       :stub-columns: 1
       :widths: 30 70

       * - Parameters
         - As many as the three terms define
       * - Control States
         - As many as the tree terms define
       * - Epoch Dependent?
         - If any of the three terms are epoch dependent

    As noted in the :class:`SeparableControlLaw`, the parameters and control
    state variables are listed in the order thy are passed to the constructor,
    in this case thrust, then mass, then orientation.
    """

    def __init__(
        self, force: ControlTerm, mass: ControlTerm, orient: ControlTerm
    ) -> None:
        super().__init__(force, mass, orient)

    @override
    def accelVec(self, t, w, varGroups, params) -> NDArray[np.double]:
        # Returns Cartesian acceleration vector
        force = self.terms[0].evalTerm(t, w, varGroups, params)
        mass = self.terms[1].evalTerm(t, w, varGroups, params)
        vec = self.terms[2].evalTerm(t, w, varGroups, params)

        return (force / mass) * vec  # type: ignore

    def _accelPartials(
        self,
        t: float,
        w: NDArray[np.double],
        varGroups: tuple[VarGroup, ...],
        params: Union[tuple[float, ...], None],
        partialFcn: str,
    ) -> NDArray[np.double]:
        # Use chain rule to combine partials of the acceleration w.r.t. some other
        #   parameter
        f = self.terms[0].evalTerm(t, w, varGroups, params)
        m = self.terms[1].evalTerm(t, w, varGroups, params)
        vec = self.terms[2].evalTerm(t, w, varGroups, params)

        dfdX = getattr(self.terms[0], partialFcn)(t, w, varGroups, params)
        dmdX = getattr(self.terms[1], partialFcn)(t, w, varGroups, params)
        dodX = getattr(self.terms[2], partialFcn)(t, w, varGroups, params)

        term1 = (vec @ dfdX / m) if dfdX.size > 0 else 0
        term2 = (-f * vec / (m * m)) @ dmdX if dmdX.size > 0 else 0
        term3 = (f / m) * dodX if dodX.size > 0 else 0

        partials = term1 + term2 + term3
        return np.asarray([]) if isinstance(partials, int) else partials

    @override
    def partials_accel_wrt_coreState(
        self, t, w, varGroups, params
    ) -> NDArray[np.double]:
        return self._accelPartials(
            t, w, varGroups, params, "partials_term_wrt_coreState"
        )

    @override
    def partials_accel_wrt_ctrlState(
        self, t, w, varGroups, params
    ) -> NDArray[np.double]:
        return self._accelPartials(
            t, w, varGroups, params, "partials_term_wrt_ctrlState"
        )

    @override
    def partials_accel_wrt_epoch(self, t, w, varGroups, params) -> NDArray[np.double]:
        return self._accelPartials(t, w, varGroups, params, "partials_term_wrt_epoch")

    @override
    def partials_accel_wrt_params(self, t, w, varGroups, params) -> NDArray[np.double]:
        return self._accelPartials(t, w, varGroups, params, "partials_term_wrt_params")
