"""
Low-thrust control objects
"""
from abc import ABC, abstractmethod

import numpy as np


class ControlTerm(ABC):
    """
    Represents a term in the control equations
    """

    def __init__(self):
        self.coreStateSize = 6  # TODO get from model
        self.paramIx0 = None

    @property
    def epochIndependent(self):
        return True

    @property
    def params(self):
        return []

    @property
    def numStates(self):
        return 0

    @property
    def stateICs(self):
        return np.zeros((self.numStates,))

    def stateDiffEqs(self, t, y, varGroups, params):
        # Differential equations governing the control states, i.e., their time
        #   derivatives
        return np.zeros((self.numStates,))

    @abstractmethod
    def evalTerm(self, t, y, varGroups, params):
        # Evaluate the control term and return it
        pass

    @abstractmethod
    def partials_term_wrt_coreState(self, t, y, varGroups, params):
        pass

    @abstractmethod
    def partials_term_wrt_ctrlState(self, t, y, varGroups, params):
        pass

    @abstractmethod
    def partials_term_wrt_epoch(self, t, y, varGroups, params):
        pass

    @abstractmethod
    def partials_term_wrt_params(self, t, y, varGroups, params):
        pass

    def partials_coreStateDEQs_wrt_ctrlState(self, t, y, varGroups, params):
        # The partial derivatives of the core STATE Diff Eqs with
        #   respect to the control state vector
        return np.zeros((self.coreStateSize, self.numStates))

    def partials_ctrlStateDEQs_wrt_coreState(self, t, y, varGroups, params):
        # The partial derivatives of the control state Diff Eqs with respect to
        #   the core state vector
        return np.zeros((self.numStates, self.coreStateSize))

    def partials_ctrlStateDEQs_wrt_ctrlState(self, t, y, varGroups, params):
        # The partial derivatives of the control state Diff Eqs with respect to
        #   the control state vector
        return np.zeros((self.numStates, self.numStates))

    def partials_ctrlState_wrt_epoch(self, t, y, varGroups, params):
        return np.zeros((self.numStates,))

    def partials_ctrlState_wrt_params(self, t, y, varGroups, params):
        return np.zeros((self.numStates, len(params)))


class ConstThrustTerm(ControlTerm):
    """
    Defines a constant thrust. The thrust is stored as a parameter
    """

    def __init__(self, thrust):
        super().__init__()
        self.thrust = thrust

    @property
    def params(self):
        return [self.thrust]

    def evalTerm(self, t, y, varGroups, params):
        return params[self.paramIx0]

    def partials_term_wrt_coreState(self, t, y, varGroups, params):
        return np.zeros((1, self.coreStateSize))

    def partials_term_wrt_ctrlState(self, t, y, varGroups, params):
        return None  # no control states

    def partials_term_wrt_epoch(self, t, y, varGroups, params):
        return np.asarray([0])

    def partials_term_wrt_params(self, t, y, varGroups, params):
        partials = np.zeros(1, len(params))
        partials[self.paramIx0] = 1
        return partials


class ConstMassTerm(ControlTerm):
    """
    Defines a constant mass. The mass is stored as a parameter
    """

    def __init__(self, mass):
        super().__init__()
        self.mass = mass

    @property
    def params(self):
        return [self.mass]

    def evalTerm(self, t, y, varGroups, params):
        return params[self.paramIx0]

    def partials_term_wrt_coreState(self, t, y, varGroups, params):
        return np.zeros((1, self.coreStateSize))

    def partials_term_wrt_ctrlState(self, t, y, varGroups, params):
        return None  # no control states

    def partials_term_wrt_epoch(self, t, y, varGroups, params):
        return np.asarray([0])

    def partials_term_wrt_params(self, t, y, varGroups, params):
        partials = np.zeros(1, len(params))
        partials[self.paramIx0] = 1
        return partials


class ConstOrientTerm(ControlTerm):
    """
    Defines a constant thrust orientation in the working frame. Orientation
    is parameterized via spherical angles alpha and beta, which are stored as
    parameters
    """

    def __init__(self, alpha, beta):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    @property
    def params(self):
        return [self.alpha, self.beta]

    def evalTerm(self, t, y, varGroups, params):
        alpha = params[self.paramIx0]
        beta = params[self.paramIx0 + 1]
        return np.asarray(
            [np.cos(beta) * np.cos(alpha), np.cos(beta) * np.sin(alpha), np.sin(beta)]
        )

    def partials_term_wrt_coreState(self, t, y, varGroups, params):
        return np.zeros((3, self.coreStateSize))

    def partials_term_wrt_ctrlState(self, t, y, varGroups, params):
        return None  # no control states

    def partials_term_wrt_epoch(self, t, y, varGroups, params):
        return np.zeros((3,))

    def partials_term_wrt_params(self, t, y, varGroups, params):
        partials = np.zeros(3, len(params))
        partials[:, self.paramIx0] = [
            -np.cos(beta) * np.sin(alpha),
            np.cos(beta) * np.cos(alpha),
            0,
        ]
        partials[:, self.paramIx0 + 1] = [
            -np.sin(beta) * np.cos(alpha),
            -np.sin(beta) * np.sin(alpha),
            np.cos(beta),
        ]
        return partials


class ControlLaw:
    """
    A control law that augments the natural acceleration acting on the body
    """

    def __init__(self, force, mass, orient):
        self.force = force
        self.mass = mass
        self.orient = orient

    @property
    def numStates(self):
        return self.force.numCore + self.mass.numCore + self.orient.numCore

    def stateICs(self):
        return np.concatenate(force.stateICs, mass.stateICs, orient.stateICs)

    def stateDiffEqs(self, t, y, varGroups, params):
        # Differential equations governing the control states, i.e., their time
        #   derivatives
        return np.concatenate(
            force.stateDiffEqs, mass.stateDiffEqs, orient.stateDiffEqs
        )

    @property
    def stateNames(self):
        # TODO
        pass

    @property
    def params(self):
        return np.concatenate(self.force.params + self.mass.params + self.orient.params)

    def registerParams(self, ix0):
        # Tells the control terms where their params live within the vector
        self.force.paramIx0 = ix0
        ix0 += len(self.force.params)

        self.mass.paramIx0 = ix0
        ix0 += len(self.mass.params)

        self.orient.paramIx0 = ix0

    def accelVec(t, y, varGroups, params):
        # Returns Cartesian acceleration vector
        force = self.force.eval(t, y, varGroups, params)
        mass = self.massPolicty.eval(t, y, varGroups, params)
        vec = self.orient.eval(t, y, varGroups, params)

        return (force / mass) * vec

    def _accelPartials(self, t, y, varGroups, params, partialFcn):
        # Use chain rule to combine partials of the acceleration w.r.t. some other
        #   parameter
        f = self.force.eval(t, y, varGroups, params)
        m = self.massPolicty.eval(t, y, varGroups, params)
        vec = self.orient.eval(t, y, varGroups, params)

        dfdX = getattr(self.force, partialFcn)(t, y, varGroups, params)
        dmdX = getattr(self.mass, partialFcn)(t, y, varGroups, params)
        dodX = getattr(self.orient, partialFcn)(t, y, varGroups, params)

        # TODO adjust for matrix notation
        return dfdX * vec / m + f * np.log(m) * vec * dmdX + f * dodX / m

    def partials_accel_wrt_coreState(self, t, y, varGroups, params):
        return self._accelPartials(
            t, y, varGroups, params, "partials_term_wrt_coreState"
        )

    def partials_accel_wrt_ctrlState(self, t, y, varGroups, params):
        return self._accelPartials(
            t, y, varGroups, params, "partials_term_wrt_ctrlState"
        )

    def partials_accel_wrt_epoch(self, t, y, varGroups, params):
        return self._accelPartials(t, y, varGroups, params, "partials_term_wrt_epoch")

    def partials_accel_wrt_params(self, t, y, varGroups, params):
        return self._accelPartials(t, y, varGroups, params, "partials_term_wrt_params")

    def partials_ctrlStateDEQs_wrt_coreState(self, t, y, varGroups, params):
        # The partial derivatives of the control state Diff Eqs with respect to
        #   the core state vector
        return np.concatenate(
            self.force.partials_ctrlStateDEQs_wrt_coreState(t, y, varGroups, params),
            self.mass.partials_ctrlStateDEQs_wrt_coreState(t, y, varGroups, params),
            self.orient.partials_ctrlStateDEQs_wrt_coreState(t, y, varGroups, params),
        )

    def partials_ctrlStateDEQs_wrt_ctrlState(self, t, y, varGroups, params):
        # The partial derivatives of the control state Diff Eqs with respect to
        #   the control state vector
        return np.concatenate(
            self.force.partials_ctrlStateDEQs_wrt_ctrlState(t, y, varGroups, params),
            self.mass.partials_ctrlStateDEQs_wrt_ctrlState(t, y, varGroups, params),
            self.orient.partials_ctrlStateDEQs_wrt_ctrlState(t, y, varGroups, params),
        )

    def partials_ctrlStateDEQs_wrt_epoch(self, t, y, varGroups, params):
        # The partial derivatives of the control state Diff Eqs with respect to
        #   the epoch
        return np.concatenate(
            self.force.partials_ctrlStateDEQs_wrt_epoch(t, y, varGroups, params),
            self.mass.partials_ctrlStateDEQs_wrt_epoch(t, y, varGroups, params),
            self.orient.partials_ctrlStateDEQs_wrt_epoch(t, y, varGroups, params),
        )

    def partials_ctrlStateDEQs_wrt_params(self, t, y, varGroups, params):
        # The partial derivatives of the control state Diff Eqs with respect to
        #   the parameters
        return np.concatenate(
            self.force.partials_ctrlStateDEQs_wrt_params(t, y, varGroups, params),
            self.mass.partials_ctrlStateDEQs_wrt_params(t, y, varGroups, params),
            self.orient.partials_ctrlStateDEQs_wrt_params(t, y, varGroups, params),
        )
