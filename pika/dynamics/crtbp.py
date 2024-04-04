"""
Circular Restricted Three Body Problem Dynamics
"""
import numpy as np
from numba import njit

from pika.data import GRAV_PARAM
from pika.dynamics import AbstractDynamicsModel, EOMVars
from pika.dynamics import ModelConfig as BaseModelConfig


class ModelConfig(BaseModelConfig):
    def __init__(self, body1, body2):
        primary = body1 if body1.gm > body2.gm else body2
        secondary = body2 if body1.gm > body2.gm else body1
        totalGM = primary.gm + secondary.gm

        super().__init__(body1, body2, mu=secondary.gm / totalGM)

        self._charL = secondary.sma
        self._charM = totalGM / GRAV_PARAM
        self._charT = np.sqrt(self._charL**3 / totalGM)


class DynamicsModel(AbstractDynamicsModel):
    def __init__(self, config):
        super().__init__(config)
        self._mu = config.params["mu"]

    @njit
    def evalEOMs(self, t, q, eomVars=EOMVars.STATE):
        qDot = np.zeros((self.stateSize(eomType),))

        # Pre-compute some values; multiplication is faster than exponents
        r13 = np.sqrt((q[0] + self._mu) * (q[0] + self._mu) + q[1] * q[1] + q[2] * q[2])
        r23 = np.sqrt(
            (q[0] - 1 + self._mu) * (q[0] - 1 + self._mu) + q[1] * q[1] + q[2] * q[2]
        )
        omm = 1 - self._mu
        r23_3 = r23 * r23 * r23
        r13_3 = r13 * r13 * r13

        # State variable derivatives
        qDot[0] = q[3]
        qDot[1] = q[4]
        qDot[2] = q[5]
        qDot[3] = (
            2 * q[4]
            + q[0]
            - omm * (q[0] + self._mu) / r13_3
            - self._mu * (q[0] - omm) / r23_3
        )
        qDot[4] = q[1] - 2 * q[3] - omm * q[1] / r13_3 - mu * q[1] / r23_3
        qDot[5] = -omm * q[2] / r13_3 - self._mu * q[2] / r23_3

        # Compute STM elements
        if EOMVars.STM in eomVars:
            r13_5 = r13_3 * r13 * r13
            r23_5 = r23_3 * r23 * r23

            # Compute the pseudopotential Jacobian
            #  U = [Uxx, Uyy, Uzz, Uxy, Uxz, Uyz]
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
            #   q[6] through q[42] represent the STM (Phi) in column-major order

            # first three rows of PhiDot are the last three rows of Phi
            for r in range(3):
                for c in range(6):
                    qDot[6 + 6 * c + r] = q[6 + 6 * c + r + 3]

            for c in range(6):
                qDot[6 + 6 * c + 3] = (
                    U[0] * q[6 + 6 * c + 0]
                    + U[3] * q[6 + 6 * c + 1]
                    + U[4] * q[6 + 6 * c + 2]
                    + 2 * q[6 + 6 * c + 4]
                )
                qDot[6 + 6 * c + 4] = (
                    U[3] * q[6 + 6 * c + 0]
                    + U[1] * q[6 + 6 * c + 1]
                    + U[5] * q[6 + 6 * c + 2]
                    - 2 * q[6 + 6 * c + 3]
                )
                qDot[6 + 6 * c + 5] = (
                    U[4] * q[6 + 6 * c + 0]
                    + U[5] * q[6 + 6 * c + 1]
                    + U[2] * q[6 + 6 * c + 2]
                )

            # There are no epoch or parameter dependencies

    # --- End of EOMs ---

    @njit
    def bodyPos(self, ix, t):
        if ix == 0:
            return np.array([-self.params["mu"], 0.0, 0.0])
        elif ix == 1:
            return np.array([1 - self.params["mu"], 0.0, 0.0])
        else:
            raise ValueError(f"Index {ix} must be zero or one")

    @njit
    def bodyVel(self, ix, t):
        if ix in [0, 1]:
            return np.zeros((3,))
        else:
            raise ValueError(f"Index {ix} must be zero or one")

    def stateSize(self, eomType):
        return 6 * (EOMVars.STATE in eomType) + 36 * (EOMVars.STM in eomType)
