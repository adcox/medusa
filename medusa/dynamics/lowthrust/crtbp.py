"""
CR3BP with Low-Thrust
=====================

This model augments the "ballistic" CR3BP with a low-thrust acceleration.
See :doc:`dynamics.crtbp` for a description of the ballistic model. As described
in the :doc:`dynamics.lowthrust` documentation, the low-thrust acceleration is
added to the ballistic acceleration, yielding the governing equations,

.. math::
   \\dot{\\vec{q}} = \\begin{Bmatrix}
     \\dot{x} \\\\ \\dot{y} \\\\ \\dot{z} \\\\
     2\\dot{y} + x - (1-\\mu)(x+\\mu)/r_{13}^3 - \\mu(x - 1 + \\mu)/r_{23}^3
         \\textcolor{orange}{+ \\vec{a}_u \\cdot \\hat{x}} \\\\
     -2\\dot{x} + y - (1-\\mu) y / r_{13}^3 - \\mu y / r_{23}^3
         \\textcolor{orange}{+ \\vec{a}_u \\cdot \\hat{y}} \\\\
     - (1-\\mu) z / r_{13}^3 - \\mu z / r_{23}^3 
         \\textcolor{orange}{+ \\vec{a}_u \\cdot \\hat{z}}
   \\end{Bmatrix}

The ballistic CR3BP does not define any parameters, so the only parameters that
are part of the model are those defined by the control law. Similarly, the CR3BP
is an epoch-independent model, so the
:math:`\\partial \\dot{\\vec{q}}_c / \\partial T` term in the epoch partials
expansion is zero; the only epoch 
dependencies that exist are those defined by the control law.

Reference
=============

.. autoclass:: DynamicsModel
   :members:
   :show-inheritance:

"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Union

import numpy as np
from numpy.typing import NDArray

from medusa import util
from medusa.data import Body
from medusa.dynamics import TVarGroup, VarGroup
from medusa.dynamics.crtbp import DynamicsModel as CrtbpDynamics, State as CrtbpState
from medusa.typing import FloatArray, override

from . import ControlLaw


class State(CrtbpState):
    @override
    def __init__(
        self,
        model: DynamicsModel,
        data: FloatArray,
        time: float = 0.0,
        center: str = "Barycenter",
        frame: str = "Rotating",
    ) -> None:
        super().__init__(model, data, time, center, frame)
        if not isinstance(model, DynamicsModel):
            raise TypeError(
                "Model must be a medusa.dynamics.lowthrust.crtbp.DynamicsModel"
            )

        # Specify more specific model type
        self.model: DynamicsModel

    @override
    def groupSize(self, varGroups: Union[TVarGroup, Sequence[TVarGroup]]) -> int:
        varGroups = util.toList(varGroups)
        ctrl = self.model.ctrlLaw
        N = 6 + ctrl.numStates
        nCtrlParam = len(ctrl.params)

        return (
            N * (VarGroup.STATE in varGroups)
            + N**2 * (VarGroup.STM in varGroups)
            + N
            * (not self.model.epochIndependent and VarGroup.EPOCH_PARTIALS in varGroups)
            + N * nCtrlParam * (VarGroup.PARAM_PARTIALS in varGroups)
        )

    @override
    def coords(self, varGroup: TVarGroup = VarGroup.STATE) -> list[str]:
        if varGroup == VarGroup.STATE:
            baseNames = super().coords(varGroup)
            return baseNames + self.model.ctrlLaw.stateNames
        else:
            return super().coords(varGroup)


class DynamicsModel(CrtbpDynamics):
    def __init__(self, body1: Body, body2: Body, ctrlLaw: ControlLaw):
        super().__init__(body1, body2)

        if not isinstance(ctrlLaw, ControlLaw):
            raise TypeError("ctrlLaw must be a ControlLaw object")

        self.ctrlLaw = ctrlLaw
        self.ctrlLaw.register(6, 0)  # set coreStateSize and paramIx0

    @property
    @override
    def epochIndependent(self) -> bool:
        return self.ctrlLaw.epochIndependent

    @property
    @override
    def params(self) -> FloatArray:
        """The default values for the model and control law parameters"""
        return self.ctrlLaw.params

    def makeState(self, data, time, center, frame) -> State:
        return State(self, data, time, center, frame)

    @override
    def diffEqs(self, t, w, varGroups, params=None) -> NDArray[np.double]:
        mu = self.massRatio

        P = len(self.ctrlLaw.params)
        if P > 0 and (params is None or not len(params) == P):
            raise ValueError(f"Expecting {P} params; {params} does not match")

        wdot = np.zeros(w.shape)

        # Pre-compute some values; multiplication is faster than exponents
        omm = 1 - mu
        r13 = np.sqrt((w[0] + mu) * (w[0] + mu) + w[1] * w[1] + w[2] * w[2])
        r23 = np.sqrt((w[0] - omm) * (w[0] - omm) + w[1] * w[1] + w[2] * w[2])
        r23_3 = r23 * r23 * r23
        r13_3 = r13 * r13 * r13

        # Base model state variable derivatives
        wdot[:3] = w[3:6]
        wdot[3] = (
            2 * w[4] + w[0] - omm * (w[0] + mu) / r13_3 - mu * (w[0] - omm) / r23_3
        )
        wdot[4] = w[1] - 2 * w[3] - omm * w[1] / r13_3 - mu * w[1] / r23_3
        wdot[5] = -omm * w[2] / r13_3 - mu * w[2] / r23_3

        # Add accel from control
        wdot[3:6] += self.ctrlLaw.accelVec(t, w, varGroups, params)[:, 0]

        count = 6  # track number of equations

        nCtrl = self.ctrlLaw.numStates
        nCore = count
        N = nCore + nCtrl  # total number of states

        # Control state variable derivatives
        if nCtrl > 0:
            wdot[count : count + nCtrl] = self.ctrlLaw.stateDiffEqs(
                t, w, varGroups, params
            )
            count += nCtrl

        # Compute STM elements
        if VarGroup.STM in varGroups:
            r13_5 = r13_3 * r13 * r13
            r23_5 = r23_3 * r23 * r23

            # Construct state Jacobian, A = dqdot/dq
            A = np.zeros((N, N))

            # derivatives of velocity terms are simple
            A[0, 3], A[1, 4], A[2, 5] = 1, 1, 1

            # Uxx = d/dx (dvx/dt)
            A[3, 0] = (
                1
                - omm / r13_3
                - mu / r23_3
                + 3 * omm * (w[0] + mu) * (w[0] + mu) / r13_5
                + 3 * mu * (w[0] - omm) * (w[0] - omm) / r23_5
            )

            # Uyy = d/dy (dvy/dt)
            A[4, 1] = (
                1
                - omm / r13_3
                - mu / r23_3
                + 3 * omm * w[1] * w[1] / r13_5
                + 3 * mu * w[1] * w[1] / r23_5
            )

            # Uzz = d/dz (dvz/dt)
            A[5, 2] = (
                -omm / r13_3
                - mu / r23_3
                + 3 * omm * w[2] * w[2] / r13_5
                + 3 * mu * w[2] * w[2] / r23_5
            )

            # Uxy = d/dy (dvx/dt) = d/dx (dvy/dt) = Uyx
            A[3, 1] = (
                3 * omm * (w[0] + mu) * w[1] / r13_5
                + 3 * mu * (w[0] - omm) * w[1] / r23_5
            )
            A[4, 0] = A[3, 1]

            # Uxz = d/dz (dvx/dt) = d/dx (dvx/dt) = Uzx
            A[3, 2] = (
                3 * omm * (w[0] + mu) * w[2] / r13_5
                + 3 * mu * (w[0] - omm) * w[2] / r23_5
            )
            A[5, 0] = A[3, 2]

            # Uyz = d/dz (dvy/dt) = d/dy (dvz/dt) = Uzy
            A[4, 2] = 3 * omm * w[1] * w[2] / r13_5 + 3 * mu * w[1] * w[2] / r23_5
            A[5, 1] = A[4, 2]

            A[3, 4] = 2  # d/dvy (dvx/dt)
            A[4, 3] = -2  # d/dvx (dvy/dt)

            # Partials of ctrl accel w.r.t. core states
            partials = self.ctrlLaw.partials_accel_wrt_coreState(
                t, w, varGroups, params
            )
            for r in range(3, nCore):
                for c in range(0, nCore):
                    A[r, c] += partials[r - 3, c]

            if nCtrl > 0:
                # Partials of ctrl state derivatives w.r.t. core and ctrl states
                partialsCore = self.ctrlLaw.partials_ctrlStateDEQs_wrt_coreState(
                    t, w, varGroups, params
                )
                partialsCtrl = self.ctrlLaw.partials_ctrlStateDEQs_wrt_ctrlState(
                    t, w, varGroups, params
                )
                for r in range(nCore, N):
                    for c in range(0, N):
                        if c < nCore:
                            A[r, c] += partialsCore[r - nCore, c]
                        else:
                            A[r, c] += partialsCtrl[r - nCore, c - nCore]

                # Partials of core state derivatives w.r.t. ctrl states
                partials = self.ctrlLaw.partials_accel_wrt_ctrlState(
                    t, w, varGroups, params
                )
                for r in range(3, nCore):
                    for c in range(nCore, N):
                        A[r, c] += partials[r - 3, c - nCore]

            # Compute STM derivative via matrix multiplication
            #   PhiDot = A * Phi
            #   w[6] through w[42] represent the STM (Phi) in row-major order
            phi = np.reshape(w[count : count + N * N], (N, N))
            wdot[count : count + N * N] += (A @ phi).flat
            count += N * N

        # Propagated epoch partials
        if not self.epochIndependent and VarGroup.EPOCH_PARTIALS in varGroups:
            # The differential equation for the epoch partials, dw/dT is
            #
            #    d/dt (dq/dT) = A @ (dq/dT) + (dwdot/dT)
            #
            # The dwdot/dT term is computed analytically by the control law; the
            # base model (CRTBP) has no dependence on epoch
            dwdot_dT = np.zeros(N)

            # contribution of accel terms
            partials = self.ctrlLaw.partials_accel_wrt_epoch(t, w, varGroups, params)
            for r in range(3, nCore):
                dwdot_dT[r] += partials[r - 3, 0]

            # contribution of ctrl states
            partials = self.ctrlLaw.partials_ctrlStateDEQs_wrt_epoch(
                t, w, varGroups, params
            )
            for r in range(nCore, N):
                dwdot_dT[r] += partials[r - nCore, 0]

            dq_dT = np.reshape(w[count : count + N], (N, 1))
            wdot[count : count + N] += (A @ dq_dT + dwdot_dT).flat
            count += N

        # Propagated param partials
        if VarGroup.PARAM_PARTIALS in varGroups and P > 0:
            # The differential equation for the parameter partials, dq/dp, is
            #
            #    d/dt (dq/dp) = A @ (dq/dp) + (dwdot/dp)
            #
            # The dwdot/dp term is computed analytically by the control law; the
            # base model (CRTBP) defines no parameters.
            dwdot_dp = np.zeros((N, P))

            # contribution of accel terms
            partials = self.ctrlLaw.partials_accel_wrt_params(t, w, varGroups, params)
            for r in range(3, nCore):
                for c in range(0, P):
                    dwdot_dp[r, c] += partials[r - 3, c]

            # contribution of ctrl states
            partials = self.ctrlLaw.partials_ctrlStateDEQs_wrt_params(
                t, w, varGroups, params
            )
            for r in range(nCore, N):
                for c in range(0, P):
                    dwdot_dp[r, c] += partials[r - nCore, c]

            dq_dp = np.reshape(w[count : count + N * P], (N, P))
            wdot[count : count + N * P] += (A @ dq_dp + dwdot_dp).flat
            count += N * P

        return wdot
