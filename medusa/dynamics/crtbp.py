"""
Circular Restricted Three Body Problem
===============================================

This module defines the circular restricted three-body problem (CR3BP or CRTBP)
dynamics.

Three bodies are modeled in the CR3BP: two massive "primaries," denoted 
:math:`P_1` and :math:`P_2` with masses :math:`m_1 \\geq m_2`, and a third body 
(:math:`P_3`) with negligible mass compared to the other two. The two massive primaries
orbit their mutual barycenter on circular paths.

A rotating frame, :math:`\\{\\hat{x}, ~\\hat{y}, ~\\hat{z}\\}`, is defined such that the 
two massive primaries lie on the :math:`\\hat{x}` axis and their angular momentum
is along :math:`\\hat{z}`; the :math:`\\hat{y}` axis completes the right-handed set.

The dynamics are characterized by a mass ratio, :math:`\\mu = m_2/(m_1 + m_2)`.
Furthermore, a characteristic length, :math:`L_*`, is defined to be the distance
between :math:`P_1` and :math:`P_2`, and a characteristic mass, 
:math:`M_* = m_1 + m_2`, is defined as the total system mass (neglecting the 
third body). A characteristic time is derived from the mean motion of the primaries,
:math:`T_* = \\sqrt{L_*^3 / (G M_*)}` where :math:`G` is the universal gravitational
constant.

.. autosummary::
   :nosignatures:

   ~medusa.dynamics.DynamicsModel.charL
   ~medusa.dynamics.DynamicsModel.charM
   ~medusa.dynamics.DynamicsModel.charT
   medusa.data.GRAV_PARAM

All quantities in the CR3BP model are nondimensionalized via these characteristic
quantities. For example, a distance of 1.1 is equal to :math:`1.1 L_*`, or 1.1 
times the mean distance between the two massive primaries. Similarly, a time
duration of :math:`2\\pi` is the amount of time for :math:`P_2` to complete one
revolution of :math:`P_1`.

Within this implementation of the CR3BP, the state vector is defined as the
6-element vector containing the Cartesian position and velocity of a body relative
to the system barycenter, evaluated in the rotating frame. 
The locations of the two massive primaries are fixed in the rotating frame on the
:math:`\\hat{x}` axis, thus their states are simple. Given
in nondimensional coordinates within the rotating frame,

.. math::
   \\vec{q}_1 &= \\begin{Bmatrix} -\\mu & 0 & 0 & 0 & 0 & 0\\end{Bmatrix}^T, \\\\
   \\vec{q}_2 &= \\begin{Bmatrix} 1-\\mu & 0 & 0 & 0 & 0 & 0\\end{Bmatrix}^T.

These vectors are available from the model via :func:`~DynamicsModel.bodyState`.

The position and velocity of the third primary are represented by the 
:data:`~medusa.dynamics.VarGroup.STATE` variables,

.. math::
    \\vec{q} = \\begin{Bmatrix} x & y & z & \\dot{x} & \\dot{y} & \\dot{z} \\end{Bmatrix}^T.

where the dot notation represents the derivative of the coordinate with respect to
the nondimensionalized time variable, :math:`\\tau`.

The differential equations that govern the motion of the third body are 
computed via :func:`~DynamicsModel.diffEqs`. The ``STATE`` variable differential
equations are

.. math::
   \\dot{\\vec{q}} = \\begin{Bmatrix}
     \\dot{x} \\\\ \\dot{y} \\\\ \\dot{z} \\\\
     2\\dot{y} + x - (1-\\mu)(x+\\mu)/r_{13}^3 - \\mu(x - 1 + \\mu)/r_{23}^3 \\\\
     -2\\dot{x} + y - (1-\\mu) y / r_{13}^3 - \\mu y / r_{23}^3 \\\\
     - (1-\\mu) z / r_{13}^3 - \\mu z / r_{23}^3
   \\end{Bmatrix}

where :math:`\\vec{r}_{13} = \\vec{r}_3 - \\vec{r}_1` is the location of the
third body relative to the first and :math:`\\vec{r}_{23} = \\vec{r}_3 - \\vec{r}_2`
locates :math:`P_3` relative to :math:`P_2`. The vector magnitudes are represented
via :math:`r_{13}` and :math:`r_{23}`, respectively.

Partial derivatives of the state derivative with respect to the initial state can
also be propagated via the :data:`~medusa.dynamics.VarGroup.STM` group.

.. math::
   \\dot{\\mathbf{\\Phi}} = \\mathbf{A} \\mathbf{\\Phi}

where :math:`\\mathbf{\\Phi}` is initialized to the 6x6 identity matrix and 
:math:`\\mathbf{A}` is the partial derivative of the state differential equations
with respect to the state vector,

.. math::
   \\mathbf{A} = \\frac{\\partial \\dot{\\vec{q}}}{\\partial \\vec{q}}
   = \\begin{bmatrix}
     0 & 0 & 0 & 1 & 0 & 0 \\\\
     0 & 0 & 0 & 0 & 1 & 0 \\\\
     0 & 0 & 0 & 0 & 0 & 1 \\\\
     \\Omega_{xx} & \\Omega_{xy} & \\Omega_{xz} & 0 & 2 & 0 \\\\
     \\Omega_{xy} & \\Omega_{yy} & \\Omega_{yz} & -2 & 2 & 0 \\\\
     \\Omega_{xz} & \\Omega_{yz} & \\Omega_{zz} & 0 & 0 & 0
   \\end{bmatrix}.

In this notation, :math:`\\Omega` is the "pseudo-potential" function,

.. math::
   \\Omega = \\frac{1-\\mu}{r_{13}} + \\frac{\\mu}{r_{23}} + \\frac{1}{2}(x^2 + y^2),

available via :func:`~DynamicsModel.pseudopotential`,
and the subscripts denote second-order partial derivatives with respect to the
position state variables,

.. math::
   \\Omega_{xx} = \\frac{\\partial^2 \\Omega}{\\partial x^2} &= 1 - \\frac{1 - \\mu}{r_{13}^3} - \\frac{\\mu}{r_{23}^3} + \\frac{3(1 - \\mu)(x + \\mu)^2}{r_{13}^5} + \\frac{3\\mu(x + \\mu - 1)^2}{r_{23}^5}\\\\
   \\Omega_{yy} = \\frac{\\partial^2 \\Omega}{\\partial y^2} &= 1 - \\frac{1 - \\mu}{r_{13}^3} - \\frac{\\mu}{r_{23}^3} + \\frac{3(1 - \\mu)y^2}{r_{13}^5} + \\frac{3\\mu y^2}{r_{23}^5}\\\\
   \\Omega_{zz} = \\frac{\\partial^2 \\Omega}{\\partial z^2} &=  - \\frac{1 - \\mu}{r_{13}^3} - \\frac{\\mu}{r_{23}^3} + \\frac{3(1 - \\mu)z^2}{r_{13}^5} + \\frac{3\\mu z^2}{r_{23}^5}\\\\
   \\Omega_{xy} = \\Omega_{yx} = \\frac{\\partial^2 \\Omega}{\\partial x \\partial y} &= \\frac{3(1 - \\mu)(x + \\mu)y}{r_{13}^5} + \\frac{3\\mu(x + \\mu - 1)y}{r_{23}^5}\\\\
   \\Omega_{xz} = \\Omega_{zx} = \\frac{\\partial^2 \\Omega}{\\partial x \\partial z} &= \\frac{3(1 - \\mu)(x + \\mu)z}{r_{13}^5} + \\frac{3\\mu(x + \\mu - 1)z}{r_{23}^5}\\\\
   \\Omega_{yz} = \\Omega_{zy} = \\frac{\\partial^2 \\Omega}{\\partial y \\partial z} &= \\frac{3(1 - \\mu)yz}{r_{13}^5} + \\frac{3\\mu yz}{r_{23}^5}


Reference
===========

.. autoclass:: DynamicsModel
   :members:

"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Union

import numpy as np
import pint
from numba import njit
from numpy.typing import NDArray

import medusa.util as util
from medusa.data import GRAV_PARAM, Body
from medusa.dynamics import DynamicsModel as BaseDynamics, State as BaseState, VarGroup
from medusa.typing import FloatArray, override
from medusa.units import LU, TU, UU


class State(BaseState):
    def __init__(
        self,
        model: DynamicsModel,
        data: FloatArray,
        time: float = 0.0,
        center: str = "Barycenter",
        frame: str = "Rotating",
    ) -> None:
        super().__init__(model, data, time, center, frame)

    @override
    def groupSize(self, varGroups: Union[VarGroup, Sequence[VarGroup]]) -> int:
        varGroups = util.toList(varGroups)
        return 6 * (VarGroup.STATE in varGroups) + 36 * (VarGroup.STM in varGroups)

    @override
    def coords(self, varGroup: VarGroup) -> list[str]:
        if varGroup == VarGroup.STATE:
            return ["x", "y", "z", "dx", "dy", "dz"]
        else:
            return super().coords(varGroup)  # defaults are fine for the rest

    @override
    def units(self, varGroup: VarGroup) -> list[pint.Unit]:
        if varGroup == VarGroup.STATE:
            return [LU, LU, LU, LU / TU, LU / TU, LU / TU]  # type: ignore[list-item]
        elif varGroup == VarGroup.STM:
            return (
                np.block(
                    [
                        [np.full((3, 3), UU), np.full((3, 3), TU)],
                        [np.full((3, 3), UU / TU), np.full((3, 3), UU)],
                    ]
                )
                .flatten()
                .tolist()
            )
        else:
            return []  # No epoch or parameter partials


class DynamicsModel(BaseDynamics):
    """
    CRTBP Dynamics Model

    Args:
        body1: one of the two primary bodies
        body2: the other primary body

    The two bodies are stored in :attr:`bodies` in order of decreassing mass.
    """

    def __init__(
        self,
        body1: Body,
        body2: Body,
    ):
        primary = body1 if body1.gm > body2.gm else body2
        secondary = body2 if body1.gm > body2.gm else body1

        totalGM = primary.gm + secondary.gm
        charL = secondary.sma
        charM = totalGM / GRAV_PARAM
        charT = np.sqrt(charL**3 / totalGM)

        self.massRatio: float = (secondary.gm / totalGM).magnitude
        super().__init__([primary, secondary], charL, charT, charM)

    def __eq__(self, other: object) -> bool:
        if super().__eq__(other):
            return self.massRatio == other.massRatio  # type: ignore
        else:
            return False

    @property
    @override
    def epochIndependent(self) -> bool:
        return True

    @property
    @override
    def params(self) -> FloatArray:
        return []

    @override
    def makeState(self, data, time, center, frame) -> State:
        return State(self, data, time, center, frame)

    @override
    def diffEqs(
        self,
        t: float,
        w: NDArray[np.double],
        varGroups: tuple[VarGroup, ...],
        params: Union[tuple[float, ...], None] = None,
    ) -> NDArray[np.double]:
        return DynamicsModel._eoms(t, w, self.massRatio, varGroups)

    @override
    def bodyState(
        self,
        ix: int,
        t: float,
        w: FloatArray = [],
        varGroups: tuple[VarGroup, ...] = (VarGroup.STATE,),
        params: Union[FloatArray, None] = None,
    ) -> NDArray[np.double]:
        if ix == 0:
            return np.asarray([-self.massRatio, 0.0, 0.0, 0.0, 0.0, 0.0])
        elif ix == 1:
            return np.asarray([1 - self.massRatio, 0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            raise IndexError(f"Index {ix} must be zero or one")

    def equilibria(self, tol: float = 1e-12) -> NDArray[np.double]:
        """
        Compute the locations of the equilibrium solutions for this model.

        The CR3BP has five equilibrium solutions, also named "Lagrange Points"
        and numbered L1, L2, L3, L4, and L5. The first three points exist
        on the rotating x-axis and are sometimes called the "colinear points."
        The L4 and L5 solutions are located at the tips of equilateral triangles
        with vertices at the two primaries.

        Args:
            tol: tolerance for the Newton-Raphson scheme

        Returns:
            a 5x3 array locating the equilibrium solutions in space.

        **Algorithm**

        The triangular points are easily computed analytically,

        .. math::
           \\vec{r}_{L4} &= \\begin{Bmatrix} 1/2 - \\mu & \\sqrt{3}/2 & 0 \\end{Bmatrix}\\\\
           \\vec{r}_{L5} &= \\begin{Bmatrix} 1/2 - \\mu & -\\sqrt{3}/2 & 0 \\end{Bmatrix}

        The colinear points cannot be located analytically and are instead located
        via a numerical Newton-Raphson scheme. All three solutions satisfy the
        equation,

        .. math::
           x - (1-\\mu)\\frac{x + \\mu}{A(x+\\mu)^3} - \\mu\\frac{x - 1 + \\mu}{B(x-1+\\mu)^3} = 0

        where :math:`A = \\mathrm{sgn}(x+\\mu)` and :math:`B = \\mathrm{sgn}(x - 1 + \\mu)`.
        To quickly solve for the location of each point, a coordinate :math:`\\gamma`
        is defined for each case and substituted into the equation above, along
        with the appropriate signs for :math:`A` and :math:`B`. The iterative
        algorithm,

        .. math::
           \\gamma_{n+1} = \\gamma_n - \\frac{f(\\gamma)}{f'(\\gamma)}

        is then used to solve for :math:`\\gamma` in each case. The iterations
        stop when :math:`\\gamma_{n+1} - \\gamma_n \\leq \\epsilon` where
        :math:`\\epsilon` is the ``tol`` value provided.

        .. list-table::
           :header-rows: 1

           * - Solution
             - Location
             - A
             - B
             - Substitution
           * - L1
             - :math:`-\\mu < x \\leq 1 - \\mu`
             - 1
             - -1
             - :math:`x = 1 - \\mu - \\gamma`
           * - L2
             - :math:`1 - \\mu < x < \\infty`
             - 1
             - 1
             - :math:`x = 1 - \\mu + \\gamma`
           * - L3
             - :math:`-\\infty < x < -\\mu`
             - -1
             - -1
             - :math:`x = -\\mu - \\gamma`
        """
        # TODO document in module docs
        mu = self.massRatio

        pts = np.asarray(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0.5 - mu, np.sin(np.pi / 3), 0],
                [0.5 - mu, -np.sin(np.pi / 3), 0],
            ]
        )

        # Solver for L1
        #   Initial guess for g from Szebehely
        g, g_prev = (mu / (3 * (1 - mu))) ** (1.0 / 3.0), -999
        while abs(g - g_prev) > tol:
            g_prev = g
            g = g - ((mu / (g**2)) - (1.0 - mu) / (1.0 - g) ** 2 - g - mu + 1.0) / (
                -2 * mu / g**3 - 2 * (1.0 - mu) / (1.0 - g) ** 3 - 1.0
            )
        pts[0, 0] = 1.0 - mu - g

        # Solver for L2
        #   Initial guess for g from Szebehely
        g, g_prev = (mu / (3 * (1 - mu))) ** (1.0 / 3.0), -999
        while abs(g - g_prev) > tol:
            g_prev = g
            g = g - (-mu / g**2 - (1 - mu) / (1 + g) ** 2 - mu + 1 + g) / (
                2 * mu / g**3 + 2 * (1 - mu) / (1 + g) ** 3 + 1
            )
        pts[1, 0] = 1.0 - mu + g

        # Solver for L3
        #   Initial guess for g from Szebehely
        g, g_prev = 1.0 - 7 * mu / 12.0, -999
        while abs(g - g_prev) > tol:
            g_prev = g
            g = g - (mu / (-1 - g) ** 2 + (1 - mu) / g**2 - mu - g) / (
                -2 * mu / (1 + g) ** 3 - 2 * (1 - mu) / g**3 - 1
            )
        pts[2, 0] = -mu - g

        return pts

    def pseudopotential(self, w: FloatArray) -> float:
        """
        Compute the pseudopotential

        .. math::
           \\Omega = \\frac{1-\\mu}{r_{13}} + \\frac{\\mu}{r_{23}} + \\frac{1}{2}(x^2 + y^2)

        Args:
            w: a state vector that includes at least the three position variables

        Returns:
            the value of the pseudopotential at the provided position
        """
        # TODO document in module docs

        mu = self.massRatio
        r13 = np.sqrt((w[0] + mu) ** 2 + w[1] ** 2 + w[2] ** 2)
        r23 = np.sqrt((w[0] - 1 + mu) ** 2 + w[1] ** 2 + w[2] ** 2)
        return (1 - mu) / r13 + mu / r23 + (w[0] ** 2 + w[1] ** 2) / 2

    def partials_pseudopot_wrt_position(self, w: FloatArray) -> NDArray[np.double]:
        """
        Compute the partial derivatives of the pseudo potential with respect to
        the position states

        Args:
            w: a state vector that includes at least the three position variables

        Returns:
            The partial derivative of the pseudopotential with respect to the
            position variables. The partial derivatives with respect to the
            velocity variables are zero.
        """
        mu = self.massRatio
        r13 = np.sqrt((w[0] + mu) ** 2 + w[1] ** 2 + w[2] ** 2)
        r23 = np.sqrt((w[0] - 1 + mu) ** 2 + w[1] ** 2 + w[2] ** 2)
        r23_3 = r23 * r23 * r23
        r13_3 = r13 * r13 * r13

        return np.asarray(
            [
                [
                    w[0]
                    - (1 - mu) * (w[0] + mu) / r13_3
                    - mu * (w[0] - 1 + mu) / r23_3,
                    w[1] - (1 - mu) * w[1] / r13_3 - mu * w[1] / r23_3,
                    -(1 - mu) * w[2] / r13_3 - mu * w[2] / r23_3,
                ]
            ]
        )

    def jacobi(self, w: FloatArray) -> float:
        """
        Compute the Jacobi constant

        .. math::
           E &= 2\\Omega - v^2\\\\
             &= 2\\frac{1-\\mu}{r_{13}} + 2\\frac{\\mu}{r_{23}} + x^2 + y^2 - \\dot{x}^2 - \\dot{y}^2 - \\dot{z}^2

        Args:
            w: a state vector that includes at least the core state (position and
                velocity) coordinates

        Returns:
            the value of the Jacobi constant at the provided state
        """
        U = self.pseudopotential(w)
        return float(2 * U - (w[3] ** 2 + w[4] ** 2 + w[5] ** 2))

    # To work with numba.njit, primitive types are enforced for most arguments
    @staticmethod
    @njit
    def _eoms(
        t: float, q: NDArray[np.double], mu: float, varGroups: tuple[VarGroup, ...]
    ) -> NDArray[np.double]:
        qdot = np.zeros(q.shape)

        # Pre-compute some values; multiplication is faster than exponents
        omm = 1 - mu
        r13 = np.sqrt((q[0] + mu) * (q[0] + mu) + q[1] * q[1] + q[2] * q[2])
        r13_3 = r13 * r13 * r13

        r23 = np.sqrt((q[0] - omm) * (q[0] - omm) + q[1] * q[1] + q[2] * q[2])
        r23_3 = r23 * r23 * r23

        # -----------------------------
        # State variable derivatives
        # -----------------------------

        # position derivative = velocity state
        qdot[:3] = q[3:6]

        # velocity derivative
        qdot[3] = (
            q[0] + 2 * q[4] - omm * (q[0] + mu) / r13_3 - mu * (q[0] - omm) / r23_3
        )
        qdot[4] = q[1] - 2 * q[3] - omm * q[1] / r13_3 - mu * q[1] / r23_3
        qdot[5] = -omm * q[2] / r13_3 - mu * q[2] / r23_3

        # Compute STM elements
        if VarGroup.STM in varGroups:
            r13_5 = r13_3 * r13 * r13
            r23_5 = r23_3 * r23 * r23

            # Compute the pseudopotential Jacobian
            Uxx = (
                1
                - omm / r13_3
                - mu / r23_3
                + 3 * omm * (q[0] + mu) * (q[0] + mu) / r13_5
                + 3 * mu * (q[0] - omm) * (q[0] - omm) / r23_5
            )
            Uyy = (
                1
                - omm / r13_3
                - mu / r23_3
                + 3 * omm * q[1] * q[1] / r13_5
                + 3 * mu * q[1] * q[1] / r23_5
            )
            Uzz = (
                -omm / r13_3
                - mu / r23_3
                + 3 * omm * q[2] * q[2] / r13_5
                + 3 * mu * q[2] * q[2] / r23_5
            )
            Uxy = (
                3 * omm * (q[0] + mu) * q[1] / r13_5
                + 3 * mu * (q[0] - omm) * q[1] / r23_5
            )
            Uxz = (
                3 * omm * (q[0] + mu) * q[2] / r13_5
                + 3 * mu * (q[0] - omm) * q[2] / r23_5
            )
            Uyz = 3 * omm * q[1] * q[2] / r13_5 + 3 * mu * q[1] * q[2] / r23_5

            # Compute STM derivative
            #   PhiDot = A * Phi
            #   q[6] through q[42] represent the STM (Phi) in row-major order

            # first three rows of PhiDot are the last three rows of Phi
            qdot[6:24] = q[24:42]

            for c in range(6):
                qdot[24 + c] = (
                    Uxx * q[6 + c] + Uxy * q[12 + c] + Uxz * q[18 + c] + 2 * q[30 + c]
                )
                qdot[30 + c] = (
                    Uxy * q[6 + c] + Uyy * q[12 + c] + Uyz * q[18 + c] - 2 * q[24 + c]
                )
                qdot[36 + c] = Uxz * q[6 + c] + Uyz * q[12 + c] + Uzz * q[18 + c]

        # There are no epoch or parameter partials
        return qdot
