"""
Dynamical Models
================

Within the context of the ``medusa`` library, a dynamical model defines a set of
differential equations that describe the motion of a celestial object such as a
planet, moon, asteroid, star, or spacecraft. This library generally assumes that
the object's **state** is described by its Cartesian position and velocity coordinates,

.. math::
    \\vec{q}(\\tau, T, \\vec{p}) =
    \\begin{Bmatrix} x & y & z & \\dot{x} & \\dot{y} & \\dot{z} \\end{Bmatrix}^T,

where :math:`\\vec{q}` is the state vector, :math:`\\tau` is the independent variable
(usually time), :math:`T` is an absolute epoch corresponding to :math:`\\tau = 0`, 
and :math:`\\vec{p}` is a set of constant parameters (e.g., engine thrust).
In this notation, the dot over the coordinates represents the derivative with 
respect to :math:`\\tau`. Some models may append more 
variables, such as a thrust vector parameterization, to this state vector.

Within the code, the dynamics model is defined via a :class:`DynamicsModel` object.
This object provides methods to compute the derivative of the state, stores the
conversion between standard and nondimensional units, and describes the locations
of all relevant bodies throughout time.

The :class:`State` object captures the actual numerical data (e.g., the position
and velocity) as well as associated metadata such as the epoch, the central body, 
and the reference frame.

.. note::
   The ``DynamicsModel`` and ``State`` classes defined in this module are
   **abstract** and cannot be instantiated on their own. Specific implementations
   define *derived* classes based on these base objects. For an example, see the
   :mod:`medusa/dynamics/crtbp` implementation of the CR3BP.

Partial Derivatives
--------------------

In addition to describing the evolution of the state vector, models may also 
describe three other "variable groups", described by :class:`VarGroup`, that 
provide extra dynamical information and are frequently
used in differential corrections and optimization processes:

State Partials
^^^^^^^^^^^^^^

The state transition matrix (STM), :math:`\\mathbf{\\Phi}`, is identified via the 
:data:`VarGroup.STM` type.

This matrix relates changes 
in :math:`\\vec{q}` at one value of :math:`\\tau` to changes in :math:`\\vec{q}`
at a different value, i.e.,

.. math::
   \\delta \\vec{q}(\\tau_2) = \\mathbf{\\Phi}(\\tau_1, \\tau_2) \\delta \\vec{q}(\\tau_1)

Deriving an analytical expression for this matrix is difficult, but its derivative
is straightforward to define. Accordingly, the matrix is propagated along with the
state vector to yield an STM that relates the end of an arc to its beginning.
The differential equation that describes this evolution is

.. math::
   \\dot{\\mathbf{\\Phi}}(\\tau_2, \\tau_1) = \\mathbf{A}(\\tau_2) \\mathbf{\\Phi}(\\tau_2, \\tau_1)

where :math:`\\mathbf{A}` is the linearization of the dynamics,

.. math::
   \\mathbf{A} = \\frac{ \\partial \\dot{\\vec{q}} }{ \\partial \\vec{q} }.

When :math:`\\tau_2 = \\tau_1`, the state transition matrix is identity; this 
serves as the initial condition for the :data:`~VarGroup.STM` type.

Epoch Partials
^^^^^^^^^^^^^^^

The epoch partials, :math:`\\partial \\vec{q} / \\partial T`, are identified via
the :data:`VarGroup.EPOCH_PARTIALS` type.

Similar to the STM, the epoch partials are numerically integrated. Because the
time along a propagated arc, :math:`\\tau`, is independent of the epoch, :math:`T`,
the order of differentiation can be switched,

.. math::
   \\frac{\\mathrm{d}}{\\mathrm{d} \\tau} \\left( \\frac{ \\partial \\vec{q} }{ \\partial T } \\right) =
   \\frac{\\partial}{\\partial T} \\left( \\frac{ \\mathrm{d} \\vec{q} }{ \\mathrm{d} \\tau } \\right) =
   \\frac{\\partial \\dot{\\vec{q}}}{\\partial T}

The right side of the equation can be derived analytically from the differential 
equations that govern the core state. The initial condition for this term is the
zero vector.


Parameter Partials
^^^^^^^^^^^^^^^^^^

The parameter partials, :math:`\\partial \\vec{q} / \\partial \\vec{q}`, are identified
via the :data:`VarGroup.PARAM_PARTIALS`, type.

Similar to the STM and epoch partials, the parameter partials are numerically 
integrated. Because the parameters are constant and independent of the integration
variable, :math:`t`, the same differentiation "trick" as the epoch partials is 
applied to yield an expression for the derivative of the parameter partials:

.. math::
   \\frac{\\mathrm{d}}{\\mathrm{d} \\tau} \\left( \\frac{ \\partial \\vec{q} }{ \\partial \\vec{p} } \\right) =
   \\frac{\\partial}{\\partial \\vec{p}} \\left( \\frac{ \\mathrm{d} \\vec{q} }{ \\mathrm{d} \\tau } \\right) =
   \\frac{\\partial \\dot{\\vec{q}}}{\\partial \\vec{p}}

The right side of the equation can be derived analytically from the differential 
equations that govern the core state. The initial condition for this term is the
zero vector.


Variable Vector
----------------

Grouped together with matrices expanded in row-major order, the variable groups
are collected into a **variable vector**, denoted by

.. math::
   \\vec{w} = \\begin{Bmatrix} 
     \\vec{q} & \\mathbf{\\Phi} & 
     \\frac{\\partial \\vec{q}}{\\partial T} & 
     \\frac{\\partial \\vec{q}}{\\partial \\vec{p}}
   \\end{Bmatrix}.

The :class:`DynamicsModel` defines the evolution of this vector
via the :func:`~DynamicsModel.diffEqs` function, which returns the
deriative of the variable vector with respect to the independent variable,

.. math::
   \\dot{\\vec{w}} = \\frac{\\mathrm{d} \\vec{w}}{\\mathrm{d} \\tau}(\\tau, T, \\vec{p})

Defining and interacting with a variable vector is accomplished via a :class:`State`
object. For example,

.. code-block:: python

   model = DynamicsModel( ... )
   state = State(model, [1, 2, 3, 4, 5, 6], epoch = 0.0, center="Earth", frame="J2000")

Assuming the model has a 6-element state (e.g., 3D position and velocity), the
:data:`VarGroup.STATE` elements are set to the values one through six. These can
be queried directly from the state,

.. code-block:: python

   >> state[:6]

   [1, 2, 3, 4, 5, 6]

or accessed via a function,

.. code-block:: python

   >> state.get(VarGroup.STATE)

   [1, 2, 3, 4, 5, 6]

Variable vector values can also be set,

.. code-block:: python

   >> state[:6] = np.arange(6)
   >> state[:6]

   [0,1,2,3,4,5]

   >> state.set(VarGroup.STATE, np.arange(6)  # same effect

Although the full variable vector can be passed to the :class:`State` cosntructor,
or inserted as described above,
it is often easier to use "default initial conditions (ICs)". These can be applied
to the vector,

.. code-block:: python

   state.fillDefaultICs( VarGroup.STM )
   # or
   state.fillDefaultICs( [VarGroup.STM, VarGroup.EPOCH_PARTIALS] )

The :class:`State` class implements vector and matrix math so that it is possible
to manipulate the data in an intuitive algebraic way,

.. code-block:: python

   state += 1       # add one to all elements
   state -= [1, 2, 3, 4, 5, 6]  # vector difference
   state *= 1.3     # multiply all elements by 1.3
   state /= [2, 2, 1, 1, 2, 2]  # element-wise division
   state @ stm      # matrix multiply the state by an STM matrix

These operations are implemented consistently with :mod:`numpy`. Additionally,
``State``s can be combined. However, these operations are only permitted if the
model, epoch, center, and frame are the same for the two states. Derived classes 
may modify this logic if it makes sense (e.g., autonomous dynamical models may not
care if the epochs are equal).

.. code-block:: python

   state1 + state2      # add two states
   state1 - state2      # subtract two states
   state1 * state2      # RUNTIME ERROR - no physical meaning
   state1 / state2      # RUNTIME ERROR - no physical meaning




The :class:`State` class provides several other methods to manipulate the data:

.. autosummary::
   :nosignatures:

   State.groupSize
   State.units
   State.coords
   State.get
   State.set
   State.fillDefaultICs
   State.relativeTo

System Properties and Parameters
--------------------------------

Additional information about the system of equations are available via **properties**
and **parameters**. Within the context of this library, a property is a constant
characteristic of the model that cannot be modified, e.g., during a differential
corrections process. A parameter, on the other hand, may be modified without
fundamentally altering the model.

Four properties are available for any dynamical model,

.. autosummary::
   :nosignatures:

   ~DynamicsModel.charL
   ~DynamicsModel.charT
   ~DynamicsModel.charM
   ~DynamicsModel.epochIndependent

By default, the abstract model class does not define any parameters and relies on
derived classes to define the :func:`params` function to provide default values of
any parameters that apply.

Units
^^^^^
_See also: :doc:`units`_

For practically all of the encoded calculus, quantities are represented via
**normalized** length, time, and mass units, :data:`~medusa.LU`, :data:`~medusa.TU`,
:data:`~medusa.MU`. To convert to and from "standard" units (km, sec, kg, etc.),
a ``State`` provides two methods:

.. autosummary::
   :nosignatures:

   State.toBaseUnits
   State.normalize

These conversions are accomplished within a unit :class:`~pint.Context` stored
in the model as :attr:`DynamicsModel.unitContext`; the context records
the conversion from the ``LU``, ``TU``, and ``MU`` units and the standard units.

Utilities
---------

Finally, a convenience method to check the partial derivatives included in the
differential equations is provided. This method is primarily for debugging purposes
but can save many hours of time when deriving and coding the partial differential
equations.

.. autosummary::
   :nosignatures:

   ~DynamicsModel.checkPartials

Implementations
---------------

The ``DynamicsModel`` defined in this module is just that -- abstract.
Concrete implementations are included in the submodules listed below.

.. toctree::
   :maxdepth: 1

   dynamics.crtbp
   dynamics.lowthrust

Reference
============

.. autoclass:: VarGroup
   :members:

.. autoclass:: DynamicsModel
   :members:

.. autoclass:: State
   :members:

"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import copy, deepcopy
from enum import IntEnum
from typing import Union

import numpy as np
import pint
from numpy.typing import NDArray

from medusa import util
from medusa.data import Body
from medusa.typing import FloatArray
from medusa.units import kg, km, sec, ureg

logger = logging.getLogger(__name__)

__all__ = [
    # base module
    "VarGroup",
    "DynamicsModel",
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


#: Type alias for variable group
TVarGroup = Union[VarGroup, int]


# TODO need a method to return all model parameters
# TODO document that method above
# TODO update low-thrust docs about parameters if needed


class DynamicsModel(ABC):
    """
    Contains the mathematics that define a dynamical model

    Args:
        bodies: one or more primary bodies
        charL: length of one :data:`medusa.units.LU` in this model
        charT: duration of one :data:`medusa.units.TU` in this model
        charM: mass of one :data:`medusa.units.MU` in this model
    """

    _registry: dict[int, str] = {}

    def __init__(
        self,
        bodies: Sequence[Body],
        charL: pint.Quantity = 1.0 * km,
        charT: pint.Quantity = 1.0 * sec,
        charM: pint.Quantity = 1.0 * kg,
    ) -> None:
        bodies = util.toList(bodies)
        if any([not isinstance(body, Body) for body in bodies]):
            raise TypeError("Expecting Body objects")

        # Copy body objects into tuple
        #: tuple[Body]: the bodies
        self.bodies = tuple(copy(body) for body in bodies)

        if not charL.check("[length]"):
            raise pint.DimensionalityError(
                charL.units, "km", str(charL.dimensionality), "[length]"
            )
        if not charT.check("[time]"):
            raise pint.DimensionalityError(
                charT.units, "sec", str(charT.dimensionality), "[time]"
            )
        if not charM.check("[mass]"):
            raise pint.DimensionalityError(
                charM.units, "kg", str(charM.dimensionality), "[mass]"
            )

        # Register the model so that we can have a unique name for the unit context
        reg = DynamicsModel._registry
        ix = len(reg)
        clsName = self.__module__ + "." + self.__class__.__name__
        reg[ix] = clsName

        self._charL = charL
        self._charT = charT
        self._charM = charM

        # Create a context with LU, TU, and MU defined according to the quantities
        #   The LU, TU, and MU units are defined in medusa.__init__ and given
        #   useful values here
        #: pint.Context: the unit context defining conversions betwen
        #: :data:`~medusa.units.LU`, :data:`~medusa.units.TU`,
        #: :data:`~medusa.units.MU`, and standard units
        self.unitContext = pint.Context(f"{clsName}.{ix}")
        ureg.add_context(self.unitContext)
        self.unitContext.redefine(f"LU = {self._charL}")
        self.unitContext.redefine(f"TU = {self._charT}")
        self.unitContext.redefine(f"MU = {self._charM}")

    def __eq__(self, other: object) -> bool:
        """
        Compare two Model objects. This can be overridden for more specific
        comparisons in derived classes
        """
        if not isinstance(other, DynamicsModel):
            return False

        if not type(self) == type(other):
            return False

        if not all([b1 == b2 for b1, b2 in zip(self.bodies, other.bodies)]):
            return False

        # TODO need to compare property dicts by value??
        return (
            self.charL == other.charL
            and self.charT == other.charT
            and self.charM == other.charM
        )

    def __repr__(self) -> str:
        out = f"<{self.__class__.__module__}.{self.__class__.__name__}: "
        for attr in ("bodies", "charL", "charT", "charM"):
            out += "\n  {!s}={!r}".format(attr, getattr(self, attr))
        out += "\n  unitContext={!r}".format(self.unitContext.name)
        out += "\n>"
        return out

    @property
    def charL(self) -> pint.Quantity:
        """Defines the length of one :data:`medusa.units.LU` in this model"""
        return self._charL

    @property
    def charT(self) -> pint.Quantity:
        """Defines the duration of one :data:`medusa.units.TU` in this model"""
        return self._charT

    @property
    def charM(self) -> pint.Quantity:
        """Defines the mass of one :data:`medusa.units.MU` in this model"""
        return self._charM

    @property
    @abstractmethod
    def epochIndependent(self) -> bool:
        """
        Returns:
            True if the differetial equations have no dependencies on epoch,
            False otherwise.
        """
        pass

    @property
    @abstractmethod
    def params(self) -> FloatArray:
        """Default values for the model parameters"""
        pass

    @abstractmethod
    def makeState(
        self, data: FloatArray, epoch: float, center: str, frame: str
    ) -> State:
        """
        Construct a state object associated with this model. The arguments are
        described by the :class:`State` constructor.
        """
        pass

    @abstractmethod
    def bodyState(
        self,
        ix: int,
        t: float,
        w: FloatArray,
        varGroups: tuple[VarGroup, ...],
        params: Union[FloatArray, None],
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
        # TODO returns a State object?
        pass

    # Because the diffEqs method can be called millions of times during a
    # propagation, narrow type definitions are required to minimize computational
    # load.
    @abstractmethod
    def diffEqs(
        self,
        t: float,
        w: NDArray[np.double],
        varGroups: tuple[VarGroup, ...],
        params: Union[tuple[float, ...], None],
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

    def checkPartials(
        self,
        w0: State,
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

        # TODO ensure tolerances are tight enough?
        prop = Propagator(dense_output=False)
        solution = prop.propagate(w0, tspan, params=params, varGroups=State.ALL_VARS)
        sol_vec = np.concatenate(
            [
                w0._extractGroup(grp, vals=solution.y[:, -1], reshape=False)
                for grp in State.ALL_VARS[1:]
            ]
        )

        wVals = w0.get(VarGroup.STATE, units=False, reshape=False)
        wCpy = copy(w0)

        # Compute state partials (STM)
        def prop_state(y):
            wCpy[: len(y)] = y
            sol = prop.propagate(wCpy, tspan, params=params, varGroups=VarGroup.STATE)
            return sol.y[:, -1]

        num_stm = numerics.derivative_multivar(
            prop_state, wVals, initStep  # type: ignore[arg-type]
        )  # wVals *will* be NDArray[np.double]

        # Compute epoch partials
        if w0.groupSize(VarGroup.EPOCH_PARTIALS) > 0:

            def prop_epoch(epoch):
                sol = prop.propagate(
                    w0,
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
        if w0.groupSize(VarGroup.PARAM_PARTIALS) > 0:

            def prop_params(p):
                sol = prop.propagate(w0, tspan, params=p, varGroups=VarGroup.STATE)
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
        varNames = np.concatenate([w0.coords(grp) for grp in State.ALL_VARS[1:]])
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


class ModelBlockCopyMixin:
    """
    A mixin class that prevents the parent class from copying stored
    :class:`DynamicsModel` objects
    """

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if isinstance(v, DynamicsModel):
                # Models should NOT be copied
                setattr(result, k, v)
            else:
                setattr(result, k, deepcopy(v, memo))
        return result


class State(ModelBlockCopyMixin):
    """
    Bundles the data required to fully represent a state vector

    Args:
        model: the model that describes the evolution of the state
        data: the numerical state values. There must be at least a "core"
            state included, i.e., the :data:`VarGroup.STATE` elements. Additional
            elements for the state, epoch, and parameter partials can also be
            provided.
        epoch: the epoch associated with the state vector
        center: the central body for the state
            .. warning:: this attribute is currently not used
        frame: the reference frame for the state
            .. warning:: this attribute is currently not used
    """

    ALL_VARS = [
        VarGroup.STATE,
        VarGroup.STM,
        VarGroup.EPOCH_PARTIALS,
        VarGroup.PARAM_PARTIALS,
    ]

    def __init__(
        self,
        model: DynamicsModel,
        data: FloatArray,
        epoch: float,
        center: str,
        frame: str,
    ) -> None:
        if not isinstance(model, DynamicsModel):
            raise TypeError(
                "Expecting 'model' to be derived from medusa.dynamics.DynamicsModel"
            )
        # TODO a body attribute??

        # Save properties
        #: the dynamical model that describes the evolution of this state
        self.model = model
        self.epoch = copy(epoch)  #: epoch associated with the state vector
        self.center = copy(center)  #: central body for the state
        self.frame = copy(frame)  #: reference frame for the state

        # Create a data vector of NaN
        N = self.groupSize(State.ALL_VARS)
        self._data = np.full((N,), np.nan, dtype=float)

        # Copy in the data
        data = np.array(data, copy=True).flatten()
        nCore = self.groupSize(VarGroup.STATE)
        if data.size < nCore:
            raise ValueError(
                f"Data must include at least the full core state ({nCore} elements)"
            )
        self._data[: data.size] = data

    def __repr__(self):
        msg = "<State "
        msg += "model = " + self.model.__class__.__name__ + ", "
        msg += ", ".join(
            [
                "{!s} = {!r}".format(attr, getattr(self, attr))
                for attr in ("center", "frame", "epoch")
            ]
        )
        msg += ",\ndata = {!r}".format(self._data)
        msg += ">"
        return msg

    def __eq__(self, obj) -> bool:
        if not isinstance(obj, State):
            return False

        return (
            obj.center == self.center
            and obj.frame == self.frame
            and obj.epoch == self.epoch
            and np.array_equal(obj._data, self._data, equal_nan=True)
        )

    def __add__(self, obj) -> State:
        # Or should this be __concat__?
        ss = copy(self)
        ss.__iadd__(obj)
        return ss

    def __iadd__(self, obj) -> State:
        # Addition of state objects is permitted if they have the same model,
        #   center, and frame
        if isinstance(obj, State):
            if not obj.model == self.model:
                raise RuntimeError("Cannot add two states with different models")
            if not obj.center == self.center:
                raise RuntimeError("Cannot add two states with different centers")
            if not obj.frame == self.frame:
                raise RuntimeError("Cannot add two states with different frames")

        if hasattr(obj, "__len__"):
            self._data[: len(obj)].__iadd__(obj)
        else:
            self._data.__iadd__(obj)

        return self

    def __contains__(self, obj) -> bool:
        return self._data.__contains__(obj)

    def __getitem__(self, indices) -> FloatArray:
        return self._data[indices]

    def __setitem__(self, indices, value) -> None:
        self._data[indices] = value

    def __len__(self) -> int:
        return len(self._data)

    def __mul__(self, obj) -> State:
        ss = copy(self)
        ss.__imul__(obj)
        return ss

    def __imul__(self, obj) -> State:
        # States cannot be multiplied together, but a state can be multiplied by
        #   a scalar or an arbitrary vector
        if isinstance(obj, State):
            raise RuntimeError("Cannot multiply states together")

        if hasattr(obj, "__len__"):
            # Allow multiplication to affect a subset of the _data elements
            #   This makes it possible to multiply a state by an STM, for example,
            #   without running into errors
            self._data[: len(obj)].__imul__(obj)
        else:
            self._data.__imul__(obj)

        return self

    # TODO implement division??

    def __matmul__(self, obj) -> State:
        ss = copy(self)
        ss.__imatmul__(obj)
        return ss

    def __imatmul__(self, obj) -> State:
        if isinstance(obj, State):
            raise RuntimeError("Cannot multiple states together")

        if hasattr(obj, "__len__"):
            # Same logic as for __imul__: allow multiplication with a subset of _data
            self._data[: len(obj)].__imatmul__(obj)
        else:
            self._data.__imatmul__(obj)
        return self

    def __neg__(self) -> State:
        ss = copy(self)
        ss._data = -ss._data
        return ss

    def __sub__(self, obj) -> State:
        ss = copy(self)
        ss.__isub__(obj)
        return ss

    def __isub__(self, obj) -> State:
        if isinstance(obj, State):
            if not obj.model == self.model:
                raise RuntimeError("Cannot add two states with different models")
            if not obj.center == self.center:
                raise RuntimeError("Cannot add subtract states with different centers")
            if not obj.frame == self.frame:
                raise RuntimeError("Cannot add subtract states with different frames")

        if hasattr(obj, "__len__"):
            self._data[: len(obj)].__isub__(obj)
        else:
            self._data.__isub__(obj)
        return self

    @abstractmethod
    def units(self, varGroup: TVarGroup = VarGroup.STATE) -> list[pint.Unit]:
        """
        Get the units for a variable group

        Args:
            varGroup: the variable group to get units for

        Returns:
            the units for the variable group in row-major order
        """
        pass

    @abstractmethod
    def groupSize(self, varGroups: Union[TVarGroup, Sequence[TVarGroup]]) -> int:
        """
        Get the size (i.e., number of elements) for one or more variable groups.

        Args:
            varGroups: describes one or more groups of variables

        Returns:
            the size of a variable array with the specified variable groups
        """
        pass

    def coords(self, varGroup: TVarGroup = VarGroup.STATE) -> list[str]:
        """
        Get names for the variables in a group.

        This implementation provides basic representations of the variables and
        can be overridden by derived classes to give more descriptive names.

        Args:
            varGroup: the variable group

        Returns:
            a list containing the names of the variables in row-major order
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

    def get(
        self,
        varGroups: Union[TVarGroup, Sequence[TVarGroup]] = VarGroup.STATE,
        units: bool = False,
        reshape=True,
    ) -> Union[NDArray, tuple[NDArray, ...]]:
        """
        Get state values for one or more variable groups

        Args:
            varGroups: the variable groups to retrieve
            units: whether or not to return the data with units (:class:`pint.Quantity`)
                or without (:class:`float`)
            reshape: whether or not to reshape matrix elements. If ``True``, variable
                groups like :data:`VarGroup.STM` are reshaped into a 2-dimensional
                array. Otherwise, data are returned in row-major order.

        Returns:
            the requested variable groups. If multiple variable groups are requested,
            the data are returned as a :class:`tuple` in the same order as requested.
        """
        varGroups = util.toList(varGroups)

        # Get raw data in 1D array
        vals = np.concatenate(
            [self._extractGroup(grp, reshape=False) for grp in varGroups]
        )

        # Add units, if applicable
        if units:
            vals = self.toBaseUnits(vals, varGroups)

        # Reshape into matrices, if applicable
        if reshape:
            vals = tuple(  # type: ignore [assignment]
                self._extractGroup(grp, vals, varGroupsIn=varGroups, reshape=True)
                for grp in varGroups
            )
            return vals[0] if len(vals) == 1 else vals

        return vals

    def set(
        self, varGroups: Union[TVarGroup, Sequence[TVarGroup]], data: FloatArray
    ) -> None:
        """
        Set the values for a variable group

        Args:
            varGroups: the variable group(s) to set
            data: the numeric data for the variable group(s)
        """
        varGroups = util.toList(varGroups)
        for grp in varGroups:
            subdata = self._extractGroup(
                grp, vals=data, varGroupsIn=varGroups, reshape=False
            )
            ix0 = sum(self.groupSize(grpBefore) for grpBefore in range(grp))
            self._data[ix0 : ix0 + self.groupSize(grp)] = subdata

    def fillDefaultICs(
        self, varsToAppend: Union[TVarGroup, Sequence[TVarGroup]]
    ) -> None:
        """
        Write the default initial conditions for the specified variable group(s)
        to the state. **This overwrites existing data!**

        Args:
            varsToAppend: the variable group(s) to append initial conditions for
        """
        varsToAppend = util.toList(varsToAppend)
        for grp in varsToAppend:
            ic = self._defaultICs(grp)
            if ic.size == 0:
                continue
            self.set(grp, ic)

    def relativeTo(self, center: str) -> State:
        """
        Get the state relative to a different center

        .. important:: not implemented yet!
        """
        # TODO this method likely needs to go from
        # current center -> default center -> requested center, but defining the
        # "default center" is not currently accomplished
        raise NotImplementedError

    def _extractGroup(
        self,
        varGroup: TVarGroup,
        vals: Union[Sequence, NDArray, None] = None,
        varGroupsIn: Union[TVarGroup, Sequence[TVarGroup]] = ALL_VARS,
        reshape: bool = True,
    ) -> NDArray[np.double]:
        """
        Extract a variable group from a sequence.

        This is a utility function; the user will call :func:`get` to extract
        data from the state by VarGroup.

        Args:
            varGroup: the variable group to extract from ``vals``
            vals: a sequence of values. If ``None``, defaults to the state ``data``
            varGroupsIn: the variable groups in ``vals``.
            reshape:

        Returns:
            the subset of ``vals`` that corresponds to the ``VarGroup``
            group. The vector elements are reshaped into a matrix if applicable.

        Raises:
            ValueError: if ``vals`` doesn't have enough elements to extract the
                requested variable groups
        """
        # TODO should this be a "public" fcn?
        if vals is None:
            vals = self._data
        vals = np.asarray(vals)

        varGroupsIn = util.toList(varGroupsIn)
        if not varGroup in varGroupsIn:
            raise RuntimeError(
                f"Requested variable group {varGroup} is not part of input set, {varGroupsIn}"
            )

        nPre = sum([self.groupSize(tp) for tp in varGroupsIn if tp < varGroup])
        sz = self.groupSize(varGroup)

        if vals.size < nPre + sz:
            raise ValueError(
                f"Need {nPre + sz} vector elements to extract {varGroup} "
                f"but vals has size {vals.size}"
            )

        nState = self.groupSize(VarGroup.STATE)
        nCol = int(sz / nState)
        _w = vals.flatten()
        if nCol > 1 and reshape:
            return np.reshape(_w[nPre : nPre + sz], (nState, nCol))
        else:
            return np.array(_w[nPre : nPre + sz])

    def _defaultICs(self, varGroup: TVarGroup) -> NDArray[np.double]:
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

    def toBaseUnits(
        self,
        vals: Union[FloatArray, None] = None,
        varGroups: Union[TVarGroup, Sequence[TVarGroup]] = ALL_VARS,
    ) -> NDArray:
        """
        Convert normalized :class:`float` data to units.

        The :attr:`DynamicsModel.charL`, :attr:`DynamicsModel.charT`, and
        :attr:`DynamicsModel.charM` in :attr:`model` define the conversion
        from the normalized ``LU``, ``TU``, and ``MU`` units to "standard" units
        like km, sec, and kg.

        Args:
            vals: an MxN array of normalized values where N is the number of variables
                in the ``varGroups``. If ``None``, the state data is converted.
            varGroups: the variable groups included in each row of ``vals``

        Returns:
            the data from ``vals`` in "standard" units
        """
        if vals is None:
            vals = self._data

        try:
            vals = np.array(vals, ndmin=2, copy=True)
        except (ValueError, pint.errors.DimensionalityError):
            raise TypeError(
                "Expecting input data to be numeric (int, float, etc.); did "
                "you mean to call normalize()?"
            )

        N = self.groupSize(varGroups)
        if not N in vals.shape:
            raise ValueError(
                f"Dimension mismatch; 'vals' has shape {vals.shape}, which does not "
                f"include variable group size, {N}"
            )

        if not vals.shape[1] == N:
            vals = vals.T

        # Convert to standard units
        units = np.concatenate([self.units(grp) for grp in util.toList(varGroups)])
        with ureg.context(self.model.unitContext.name):  # type: ignore[arg-type]
            scale = np.full((N, N), ureg.Quantity(0.0), dtype=pint.Quantity)
            for ix in range(N):
                scale[ix, ix] = ureg.Quantity(1.0, units[ix]).to_base_units()

        return vals @ scale

    def normalize(
        self,
        vals: Union[Sequence[pint.Quantity], NDArray, None] = None,
        varGroups: Union[TVarGroup, Sequence[TVarGroup]] = ALL_VARS,
    ) -> NDArray[np.double]:
        """
        Convert :class:`~pint.Quantity` data to normalized floats in the unit
        context of this model.


        The :attr:`DynamicsModel.charL`, :attr:`DynamicsModel.charT`, and
        :attr:`DynamicsModel.charM` in :attr:`model` define the conversion
        from the normalized ``LU``, ``TU``, and ``MU`` units to "standard" units
        like km, sec, and kg.

        Args:
            vals: an MxN array of Quantities where N is the number of variables
                in the ``varGroups``. If ``None``, the state data is used.
            varGroups: the variable groups included in each row of ``vals``

        Returns:
            the data from ``vals`` in normalized units. The data are returned as
            floats instead of as Quantity objects with ``LU``, ``TU``, and ``MU``
            units.

        See also:
            :func:`units`: this function provides the normalized units for each variable
            in terms of the :data:`~medusa.units.LU`, :data:`~medusa.units.TU`, and
            :data:`~medusa.units.MU` units.
        """
        if vals is None:
            vals = self._data

        vals = np.array(vals, ndmin=2, copy=True, dtype=pint.Quantity)
        if not all(isinstance(v, pint.Quantity) for v in vals.flat):
            raise TypeError(
                "Expecting input data to be Quantities; did you mean to call toBaseUnits()?"
            )

        N = self.groupSize(varGroups)
        if not N in vals.shape:
            raise ValueError(
                f"Dimension mismatch; 'vals' has shape {vals.shape}, which does not "
                f"include variable group size, {N}"
            )

        if not vals.shape[1] == N:
            vals = vals.T

        # Convert to floats, normalizing by LU, TU, and DU
        units = np.concatenate([self.units(grp) for grp in util.toList(varGroups)])
        with ureg.context(self.model.unitContext.name):  # type: ignore[arg-type]
            scale = np.zeros((N, N))
            for ix in range(N):
                scale[ix, ix] = (
                    1.0 / ureg.Quantity(1.0, units[ix]).to_base_units().magnitude
                )

            vals = np.asarray(
                [[val.to_base_units().magnitude for val in subvals] for subvals in vals]
            )

            return vals @ scale
