"""
Circular Restricted Three Body Problem Dynamics
===============================================

This module defines the circular restricted three-body problem (CR3BP or CRTBP)
dynamics.

Three bodies are modeled in the CR3BP: two massive "primaries," denoted 
:math:`P_1` and :math:`P_2` with masses :math:`m_1 \geq m_2`, and a third body 
(:math:`P_3`) with negligible mass compared to the other two. The two massive primaries
orbit their mutual barycenter on circular paths.

A rotating frame, :math:`\{\hat{x}, ~\hat{y}, ~\hat{z}\}`, is defined such that the 
two massive primaries lie on the :math:`\hat{x}` axis and their angular momentum
is along :math:`\hat{z}`; the :math:`\hat{y}` axis completes the right-handed set.

The dynamics are characterized by a mass ratio, :math:`\mu = m_2/(m_1 + m_2)`.
Furthermore, a characteristic length, :math:`L_*`, is defined to be the distance
between :math:`P_1` and :math:`P_2`, and a characteristic mass, 
:math:`M_* = m_1 + m_2`, is defined as the total system mass (neglecting the 
third body). A characteristic time is derived from the mean motion of the primaries,
:math:`T_* = \sqrt{L_*^3 / (G M_*)}` where :math:`G` is the universal gravitational
constant.

.. autosummary::
   :nosignatures:

   ~medusa.dynamics.AbstractDynamicsModel.charL
   ~medusa.dynamics.AbstractDynamicsModel.charM
   ~medusa.dynamics.AbstractDynamicsModel.charT
   medusa.data.GRAV_PARAM

All quantities in the CR3BP model are nondimensionalized via these characteristic
quantities. For example, a distance of 1.1 is equal to :math:`1.1 L_*`, or 1.1 
times the mean distance between the two massive primaries. Similarly, a time
duration of :math:`2\pi` is the amount of time for :math:`P_2` to complete one
revolution of :math:`P_1`.

Within this implementation of the CR3BP, the state vector is defined as the
6-element vector containing the Cartesian position and velocity of a body relative
to the system barycenter, evaluated in the rotating frame. 
The locations of the two massive primaries are fixed in the rotating frame on the
:math:`\hat{x}` axis, thus their states are simple. Given
in nondimensional coordinates within the rotating frame,

.. math::
   \\vec{q}_1 &= \\begin{Bmatrix} -\mu & 0 & 0 & 0 & 0 & 0\\end{Bmatrix}^T, \\\\
   \\vec{q}_2 &= \\begin{Bmatrix} 1-\mu & 0 & 0 & 0 & 0 & 0\\end{Bmatrix}^T.

These vectors are available from the model via :func:`~DynamicsModel.bodyState`.

The position and velocity of the third primary are represented by the 
:data:`~medusa.dynamics.VarGroup.STATE` variables,

.. math::
    \\vec{q} = \\begin{Bmatrix} x & y & z & \dot{x} & \dot{y} & \dot{z} \\end{Bmatrix}^T.

where the dot notation represents the derivative of the coordinate with respect to
the nondimensionalized time variable, :math:`\\tau`.

The differential equations that govern the motion of the third body are 
computed via :func:`~DynamicsModel.diffEqs`. The ``STATE`` variable differential
equations are

.. math::
   \dot{\\vec{q}} = \\begin{Bmatrix}
     \dot{x} \\\\ \dot{y} \\\\ \dot{z} \\\\
     2\dot{y} + x - (1-\mu)(x+\mu)/r_{13}^3 - \mu(x - 1 + \mu)/r_{23}^3 \\\\
     -2\dot{x} + y - (1-\mu) y / r_{13}^3 - \mu y / r_{23}^3 \\\\
     - (1-\mu) z / r_{13}^3 - \mu z / r_{23}^3
   \\end{Bmatrix}

where :math:`\\vec{r}_{13} = \\vec{r}_3 - \\vec{r}_1` is the location of the
third body relative to the first and :math:`\\vec{r}_{23} = \\vec{r}_3 - \\vec{r}_2`
locates :math:`P_3` relative to :math:`P_2`. The vector magnitudes are represented
via :math:`r_{13}` and :math:`r_{23}`, respectively.

Partial derivatives of the state derivative with respect to the initial state can
also be propagated via the :data:`~medusa.dynamics.VarGroup.STM` group.

.. math::
   \dot{\mathbf{\Phi}} = \mathbf{A} \mathbf{\Phi}

where :math:`\mathbf{\Phi}` is initialized to the 6x6 identity matrix and 
:math:`\mathbf{A}` is the partial derivative of the state differential equations
with respect to the state vector,

.. math::
   \mathbf{A} = \\frac{\partial \dot{\\vec{q}}}{\partial \\vec{q}}
   = \\begin{bmatrix}
     0 & 0 & 0 & 1 & 0 & 0 \\\\
     0 & 0 & 0 & 0 & 1 & 0 \\\\
     0 & 0 & 0 & 0 & 0 & 1 \\\\
     \Omega_{xx} & \Omega_{xy} & \Omega_{xz} & 0 & 2 & 0 \\\\
     \Omega_{xy} & \Omega_{yy} & \Omega_{yz} & -2 & 2 & 0 \\\\
     \Omega_{xz} & \Omega_{yz} & \Omega_{zz} & 0 & 0 & 0
   \\end{bmatrix}.

In this notation, :math:`\Omega` is the "pseudo-potential" function,

.. math::
   \Omega = \\frac{1-\mu}{r_{13}} + \\frac{\mu}{r_{23}} + \\frac{1}{2}(x^2 + y^2),

and the subscripts denote second-order partial derivatives with respect to the
position state variables,

.. math::
   \Omega_{xx} = \\frac{\partial^2 \Omega}{\partial x^2} &= 1 - \\frac{1 - \mu}{r_{13}^3} - \\frac{\mu}{r_{23}^3} + \\frac{3(1 - \mu)(x + \mu)^2}{r_{13}^5} + \\frac{3\mu(x + \mu - 1)^2}{r_{23}^5}\\\\
   \Omega_{yy} = \\frac{\partial^2 \Omega}{\partial y^2} &= 1 - \\frac{1 - \mu}{r_{13}^3} - \\frac{\mu}{r_{23}^3} + \\frac{3(1 - \mu)y^2}{r_{13}^5} + \\frac{3\mu y^2}{r_{23}^5}\\\\
   \Omega_{zz} = \\frac{\partial^2 \Omega}{\partial z^2} &=  - \\frac{1 - \mu}{r_{13}^3} - \\frac{\mu}{r_{23}^3} + \\frac{3(1 - \mu)z^2}{r_{13}^5} + \\frac{3\mu z^2}{r_{23}^5}\\\\
   \Omega_{xy} = \Omega_{yx} = \\frac{\partial^2 \Omega}{\partial x \partial y} &= \\frac{3(1 - \mu)(x + \mu)y}{r_{13}^5} + \\frac{3\mu(x + \mu - 1)y}{r_{23}^5}\\\\
   \Omega_{xz} = \Omega_{zx} = \\frac{\partial^2 \Omega}{\partial x \partial z} &= \\frac{3(1 - \mu)(x + \mu)z}{r_{13}^5} + \\frac{3\mu(x + \mu - 1)z}{r_{23}^5}\\\\
   \Omega_{yz} = \Omega_{zy} = \\frac{\partial^2 \Omega}{\partial y \partial z} &= \\frac{3(1 - \mu)yz}{r_{13}^5} + \\frac{3\mu yz}{r_{23}^5}

Reference
===========

.. autoclass:: DynamicsModel
   :members:

"""
from collections.abc import Sequence
from typing import Union

import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

import medusa.util as util
from medusa.data import GRAV_PARAM, Body
from medusa.dynamics import AbstractDynamicsModel, VarGroup
from medusa.typing import FloatArray, override


class DynamicsModel(AbstractDynamicsModel):
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

        super().__init__(primary, secondary, mu=secondary.gm / totalGM)

        self._charL = secondary.sma
        self._charM = totalGM / GRAV_PARAM
        self._charT = np.sqrt(self._charL**3 / totalGM)

    @property
    @override
    def epochIndependent(self) -> bool:
        return True

    @override
    def diffEqs(
        self,
        t: float,
        w: NDArray[np.double],
        varGroups: tuple[VarGroup, ...],
        params: Union[tuple[float, ...], None] = None,
    ) -> NDArray[np.double]:
        return DynamicsModel._eoms(t, w, self._properties["mu"], varGroups)

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
            return np.asarray([-self._properties["mu"], 0.0, 0.0, 0.0, 0.0, 0.0])
        elif ix == 1:
            return np.asarray([1 - self._properties["mu"], 0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            raise IndexError(f"Index {ix} must be zero or one")

    @override
    def groupSize(self, varGroups: Union[VarGroup, Sequence[VarGroup]]) -> int:
        varGroups = util.toList(varGroups)
        return 6 * (VarGroup.STATE in varGroups) + 36 * (VarGroup.STM in varGroups)

    @override
    def varNames(self, varGroup: VarGroup) -> list[str]:
        if varGroup == VarGroup.STATE:
            return ["x", "y", "z", "dx", "dy", "dz"]
        else:
            return super().varNames(varGroup)  # defaults are fine for the rest

    # TODO njit?
    # TODO cache?
    def equilibria(self, tol: float = 1e-12) -> NDArray[np.double]:
        """
        Get equilibrium points

        TODO document in module docs
        """
        mu = self._properties["mu"]

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

    # TODO njit?
    def pseudopotential(self, w: FloatArray) -> float:
        """
        Compute the pseudopotential

        TODO document in module docs
        """
        mu = self._properties["mu"]
        r13 = np.sqrt((w[0] + mu) ** 2 + w[1] ** 2 + w[2] ** 2)
        r23 = np.sqrt((w[0] - 1 + mu) ** 2 + w[1] ** 2 + w[2] ** 2)
        return (1 - mu) / r13 + mu / r23 + (w[0] ** 2 + w[1] ** 2) / 2

    def partials_pseudopot_wrt_position(self, w: FloatArray) -> NDArray[np.double]:
        """
        Compute the partial derivatives of the pseudo potential with respect to
        the position states
        """
        mu = self._properties["mu"]
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

    # TODO njit?
    def jacobi(self, w: FloatArray) -> float:
        """
        Compute the Jacobi constant

        TODO document in module docs
        """
        # TODO unit test
        U = self.pseudopotential(w)
        return 2 * U - (w[3] ** 2 + w[4] ** 2 + w[5] ** 2)

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
