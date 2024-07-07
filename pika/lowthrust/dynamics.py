"""
Low-thrust-enabled dynamics models
"""
import numpy as np

from pika import util
from pika.crtbp import DynamicsModel as CrtbpDynamics
from pika.dynamics import VarGroups

from .control import ControlLaw


class LowThrustCrtbpDynamics(CrtbpDynamics):
    def __init__(self, body1, body2, ctrlLaw):
        super().__init__(body1, body2)

        if not isinstance(ctrlLaw, ControlLaw):
            raise TypeError("ctrlLaw must be a ControlLaw object")

        self.ctrlLaw = ctrlLaw

        # Tell the control params where they exist in the param vector
        self.ctrlLaw.registerParams(0)

    @property
    def epochIndependent(self):
        return self.ctrlLaw.epochIndependent

    def stateSize(self, varGroups):
        varGroups = util.toList(varGroups)
        N = 6 + self.ctrlLaw.numStates
        nCtrlParam = len(self.ctrlLaw.params)

        return (
            N * (VarGroups.STATE in varGroups)
            + N**2 * (VarGroups.STM in varGroups)
            + N * (not self.epochIndependent and VarGroups.EPOCH_PARTIALS in varGroups)
            + N * nCtrlParam * (VarGroups.PARAM_PARTIALS in varGroups)
        )

    def varNames(self, varGroups):
        if varGroups == VarGroups.STATE:
            baseNames = super().varNames(varGroups)
            return baseNames + self.law.stateNames
        else:
            return super().varNames(varGroups)

    def diffEqs(self, t, q, varGroups, params=None):
        mu = self._properties["mu"]

        P = len(self.ctrlLaw.params)
        if P > 0 and (params is None or not len(params) == P):
            raise ValueError(f"Expecting {P} params; {params} does not match")

        qdot = np.zeros(q.shape)

        # Pre-compute some values; multiplication is faster than exponents
        r13 = np.sqrt((q[0] + mu) * (q[0] + mu) + q[1] * q[1] + q[2] * q[2])
        r23 = np.sqrt((q[0] - 1 + mu) * (q[0] - 1 + mu) + q[1] * q[1] + q[2] * q[2])
        omm = 1 - mu
        r23_3 = r23 * r23 * r23
        r13_3 = r13 * r13 * r13

        # Base model state variable derivatives
        qdot[:3] = q[3:6]
        qdot[3] = (
            2 * q[4] + q[0] - omm * (q[0] + mu) / r13_3 - mu * (q[0] - omm) / r23_3
        )
        qdot[4] = q[1] - 2 * q[3] - omm * q[1] / r13_3 - mu * q[1] / r23_3
        qdot[5] = -omm * q[2] / r13_3 - mu * q[2] / r23_3

        # Add accel from control
        qdot[3:6] += self.ctrlLaw.accelVec(t, q, varGroups, params)[:, 0]

        count = 6  # track number of equations

        nCtrl = self.ctrlLaw.numStates
        nCore = count
        N = nCore + nCtrl  # total number of states

        # Control state variable derivatives
        if nCtrl > 0:
            qdot[count : count + nCtrl] = self.ctrlLaw.stateDiffEqs(
                t, q, varGroups, params
            )
            count += nCtrl

        # Compute STM elements
        if VarGroups.STM in varGroups:
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
                + 3 * omm * (q[0] + mu) * (q[0] + mu) / r13_5
                + 3 * mu * (q[0] - omm) * (q[0] - omm) / r23_5
            )

            # Uyy = d/dy (dvy/dt)
            A[4, 1] = (
                1
                - omm / r13_3
                - mu / r23_3
                + 3 * omm * q[1] * q[1] / r13_5
                + 3 * mu * q[1] * q[1] / r23_5
            )

            # Uzz = d/dz (dvz/dt)
            A[5, 2] = (
                -omm / r13_3
                - mu / r23_3
                + 3 * omm * q[2] * q[2] / r13_5
                + 3 * mu * q[2] * q[2] / r23_5
            )

            # Uxy = d/dy (dvx/dt) = d/dx (dvy/dt) = Uyx
            A[3, 1] = (
                3 * omm * (q[0] + mu) * q[1] / r13_5
                + 3 * mu * (q[0] - omm) * q[1] / r23_5
            )
            A[4, 0] = A[3, 1]

            # Uxz = d/dz (dvx/dt) = d/dx (dvx/dt) = Uzx
            A[3, 2] = (
                3 * omm * (q[0] + mu) * q[2] / r13_5
                + 3 * mu * (q[0] - omm) * q[2] / r23_5
            )
            A[5, 0] = A[3, 2]

            # Uyz = d/dz (dvy/dt) = d/dy (dvz/dt) = Uzy
            A[4, 2] = 3 * omm * q[1] * q[2] / r13_5 + 3 * mu * q[1] * q[2] / r23_5
            A[5, 1] = A[4, 2]

            A[3, 4] = 2  # d/dvy (dvx/dt)
            A[4, 3] = -2  # d/dvx (dvy/dt)

            # Partials of ctrl accel w.r.t. core states
            partials = self.ctrlLaw.partials_accel_wrt_coreState(
                t, q, varGroups, params
            )
            for r in range(3, nCore):
                for c in range(0, nCore):
                    A[r, c] += partials[r - 3, c]

            if nCtrl > 0:
                # Partials of ctrl state derivatives w.r.t. core and ctrl states
                partialsCore = self.ctrlLaw.partials_ctrlStateDEQs_wrt_coreState(
                    t, q, varGroups, params
                )
                partialsCtrl = self.ctrlLaw.partials_ctrlStateDEQs_wrt_ctrlState(
                    t, q, varGroups, params
                )
                for r in range(nCore, N):
                    for c in range(0, N):
                        if c < nCore:
                            A[r, c] += partialsCore[r - nCore, c]
                        else:
                            A[r, c] += partialsCtrl[r - nCore, c - nCore]

                # Partials of core state derivatives w.r.t. ctrl states
                partials = self.ctrlLaw.partials_accel_wrt_ctrlState(
                    t, q, varGroups, params
                )
                for r in range(3, nCore):
                    for c in range(nCore, N):
                        A[r, c] += partials[r - 3, c - nCore]

            # Compute STM derivative via matrix multiplication
            #   PhiDot = A * Phi
            #   q[6] through q[42] represent the STM (Phi) in row-major order
            phi = np.reshape(q[count : count + N * N], (N, N))
            qdot[count : count + N * N] += (A @ phi).flat
            count += N * N

        # Propagated epoch partials
        if not self.epochIndependent and VarGroups.EPOCH_PARTIALS in varGroups:
            # The differential equation for the epoch partials, dq/dT is
            #
            #    d/dt (dq/dT) = A @ (dq/dT) + (dqdot/dT)
            #
            # The dqdot/dT term is computed analytically by the control law; the
            # base model (CRTBP) has no dependence on epoch
            dqdot_dT = np.zeros(N)

            # contribution of accel terms
            partials = self.ctrlLaw.partials_accel_wrt_epoch(t, q, varGroups, params)
            for r in range(3, nCore):
                dqdot_dT[r] += partials[r - 3, 0]

            # contribution of ctrl states
            partials = self.ctrlLaw.partials_ctrlStateDEQs_wrt_epoch(
                t, q, varGroups, params
            )
            for r in range(nCore, N):
                dqdot_dT[r] += partials[r - nCore, 0]

            dq_dT = np.reshape(q[count : count + N], (N, 1))
            qdot[count : count + N] += (A @ dq_dT + dqdot_dT).flat
            count += N

        # Propagated param partials
        if VarGroups.PARAM_PARTIALS in varGroups and P > 0:
            # The differential equation for the parameter partials, dq/dp, is
            #
            #    d/dt (dq/dp) = A @ (dq/dp) + (dqdot/dp)
            #
            # The dqdot/dp term is computed analytically by the control law; the
            # base model (CRTBP) defines no parameters.
            dqdot_dp = np.zeros((N, P))

            # contribution of accel terms
            partials = self.ctrlLaw.partials_accel_wrt_params(t, q, varGroups, params)
            for r in range(3, nCore):
                for c in range(0, P):
                    dqdot_dp[r, c] += partials[r - 3, c]

            # contribution of ctrl states
            partials = self.ctrlLaw.partials_ctrlStateDEQs_wrt_params(
                t, q, varGroups, params
            )
            for r in range(nCore, N):
                for c in range(0, P):
                    dqdot_dp[r, c] += partials[r - nCore, c]

            dq_dp = np.reshape(q[count : count + N * P], (N, P))
            qdot[count : count + N * P] += (A @ dq_dp + dqdot_dp).flat
            count += N * P

        return qdot
