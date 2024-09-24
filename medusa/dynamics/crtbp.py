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
Furthermore, a characteristic length, :math:`L_*` is defined to be the distance
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

Within this implementation of the CR3BP, the "state" vector is defined as the
6-element vector containing the Cartesian position and velocity of a body relative
to the system barycenter, evaluated in the rotating frame.

The locations of the two massive primaries are fixed in the rotating frame on the
:math:`\hat{x}` axis, thus their states are simple. Given
in nondimensional coordinates within the rotating frame,

.. math::
   \\vec{q}_1 &= \\begin{Bmatrix} -\mu & 0 & 0 & 0 & 0 & 0\\end{Bmatrix}, \\\\
   \\vec{q}_2 &= \\begin{Bmatrix} 1-\mu & 0 & 0 & 0 & 0 & 0\\end{Bmatrix}.

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
   \dot{\Phi} = \mathbf{A} \dot{\Phi}

where :math:`\Phi` is initialized to the 6x6 identity matrix and 

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

In this notation, :math:`\Omega` is the "pseudo-potential" function and the
subscripts denote second-order partial derivatives,

.. math::
   \Omega &= \\frac{1-\mu}{r_{13}} + \\frac{\mu}{r_{23}} + \\frac{1}{2}(x^2 + y^2)\\\\
   \Omega_{xx} = \\frac{\partial^2 \Omega}{\partial x^2} &= 1 - \\frac{1 - \mu}{r_{13}^3} - \\frac{\mu}{r_{23}^3} + \\frac{3(1 - \mu)(x + \mu)^2}{r_{13}^5} + \\frac{3\mu(x + \mu - 1)^2}{r_{23}^5}\\\\
   \Omega_{yy} = \\frac{\partial^2 \Omega}{\partial y^2} &= 1 - \\frac{1 - \mu}{r_{13}^3} - \\frac{\mu}{r_{23}^3} + \\frac{3(1 - \mu)y^2}{r_{13}^5} + \\frac{3\mu y^2}{r_{23}^5}\\\\
   \Omega_{zz} = \\frac{\partial^2 \Omega}{\partial z^2} &=  - \\frac{1 - \mu}{r_{13}^3} - \\frac{\mu}{r_{23}^3} + \\frac{3(1 - \mu)z^2}{r_{13}^5} + \\frac{3\mu z^2}{r_{23}^5}\\\\
   \Omega_{xy} = \Omega_{yx} = \\frac{\partial^2 \Omega}{\partial x \partial y} &= \\frac{3(1 - \mu)(x + \mu)y}{r_{13}^5} + \\frac{3\mu(x + \mu - 1)y}{r_{23}^5}\\\\
   \Omega_{xz} = \Omega_{zx} = \\frac{\partial^2 \Omega}{\partial x \partial z} &= \\frac{3(1 - \mu)(x + \mu)z}{r_{13}^5} + \\frac{3\mu(x + \mu - 1)z}{r_{23}^5}\\\\
   \Omega_{yz} = \Omega_{zy} = \\frac{\partial^2 \Omega}{\partial y \partial z} &= \\frac{3(1 - \mu)yz}{r_{13}^5} + \\frac{3\mu yz}{r_{23}^5}

Reference
---------

.. autoclass:: DynamicsModel
   :members:
   :show-inheritance:

"""
import numpy as np
from numba import njit

import medusa.util as util
from medusa.data import GRAV_PARAM
from medusa.dynamics import AbstractDynamicsModel, VarGroup


class DynamicsModel(AbstractDynamicsModel):
    """
    CRTBP Dynamics Model

    Args:
        body1 (Body): one of the two primary bodies
        body2 (Body): the othe primary body

    The two bodies are stored in :attr:`bodies` in order of decreassing mass
    """

    def __init__(self, body1, body2):
        primary = body1 if body1.gm > body2.gm else body2
        secondary = body2 if body1.gm > body2.gm else body1
        totalGM = primary.gm + secondary.gm

        super().__init__(primary, secondary, mu=secondary.gm / totalGM)

        self._charL = secondary.sma
        self._charM = totalGM / GRAV_PARAM
        self._charT = np.sqrt(self._charL**3 / totalGM)

    @property
    def epochIndependent(self):
        return True

    def diffEqs(self, t, q, varGroups, params=None):
        return DynamicsModel._eoms(t, q, self._properties["mu"], varGroups)

    def bodyState(self, ix, t, params=None):
        if ix == 0:
            return np.asarray([-self._properties["mu"], 0.0, 0.0, 0.0, 0.0, 0.0])
        elif ix == 1:
            return np.asarray([1 - self._properties["mu"], 0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            raise IndexError(f"Index {ix} must be zero or one")

    def stateSize(self, varGroups):
        varGroups = util.toList(varGroups)
        return 6 * (VarGroup.STATE in varGroups) + 36 * (VarGroup.STM in varGroups)

    def varNames(self, varGroups):
        if varGroups == VarGroup.STATE:
            return ["x", "y", "z", "dx", "dy", "dz"]
        else:
            return super().varNames(varGroups)  # defaults are fine for the rest

    @staticmethod
    @njit
    def _eoms(t, q, mu, varGroups):
        qdot = np.zeros(q.shape)

        # Pre-compute some values; multiplication is faster than exponents
        r13 = np.sqrt((q[0] + mu) * (q[0] + mu) + q[1] * q[1] + q[2] * q[2])
        r23 = np.sqrt((q[0] - 1 + mu) * (q[0] - 1 + mu) + q[1] * q[1] + q[2] * q[2])
        omm = 1 - mu
        r23_3 = r23 * r23 * r23
        r13_3 = r13 * r13 * r13

        # State variable derivatives
        qdot[:3] = q[3:6]
        qdot[3] = (
            2 * q[4] + q[0] - omm * (q[0] + mu) / r13_3 - mu * (q[0] - omm) / r23_3
        )
        qdot[4] = q[1] - 2 * q[3] - omm * q[1] / r13_3 - mu * q[1] / r23_3
        qdot[5] = -omm * q[2] / r13_3 - mu * q[2] / r23_3

        # Compute STM elements
        if VarGroup.STM in varGroups:
            r13_5 = r13_3 * r13 * r13
            r23_5 = r23_3 * r23 * r23

            # Compute the pseudopotential Jacobian
            #   U = [Uxx, Uyy, Uzz, Uxy, Uxz, Uyz]
            U = np.zeros((6,))

            U[0] = (
                1
                - omm / r13_3
                - mu / r23_3
                + 3 * omm * (q[0] + mu) * (q[0] + mu) / r13_5
                + 3 * mu * (q[0] - omm) * (q[0] - omm) / r23_5
            )
            U[1] = (
                1
                - omm / r13_3
                - mu / r23_3
                + 3 * omm * q[1] * q[1] / r13_5
                + 3 * mu * q[1] * q[1] / r23_5
            )
            U[2] = (
                -omm / r13_3
                - mu / r23_3
                + 3 * omm * q[2] * q[2] / r13_5
                + 3 * mu * q[2] * q[2] / r23_5
            )
            U[3] = (
                3 * omm * (q[0] + mu) * q[1] / r13_5
                + 3 * mu * (q[0] - omm) * q[1] / r23_5
            )
            U[4] = (
                3 * omm * (q[0] + mu) * q[2] / r13_5
                + 3 * mu * (q[0] - omm) * q[2] / r23_5
            )
            U[5] = 3 * omm * q[1] * q[2] / r13_5 + 3 * mu * q[1] * q[2] / r23_5

            # Compute STM derivative
            #   PhiDot = A * Phi
            #   q[6] through q[42] represent the STM (Phi) in row-major order

            # first three rows of PhiDot are the last three rows of Phi
            for r in range(3):
                for c in range(6):
                    qdot[6 + 6 * r + c] = q[6 + 6 * (r + 3) + c]

            for c in range(6):
                qdot[6 + 6 * 3 + c] = (
                    U[0] * q[6 + 6 * 0 + c]
                    + U[3] * q[6 + 6 * 1 + c]
                    + U[4] * q[6 + 6 * 2 + c]
                    + 2 * q[6 + 6 * 4 + c]
                )
                qdot[6 + 6 * 4 + c] = (
                    U[3] * q[6 + 6 * 0 + c]
                    + U[1] * q[6 + 6 * 1 + c]
                    + U[5] * q[6 + 6 * 2 + c]
                    - 2 * q[6 + 6 * 3 + c]
                )
                qdot[6 + 6 * 5 + c] = (
                    U[4] * q[6 + 6 * 0 + c]
                    + U[5] * q[6 + 6 * 1 + c]
                    + U[2] * q[6 + 6 * 2 + c]
                )

        # There are no epoch or parameter partials
        return qdot
