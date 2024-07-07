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
        if self.numStates == 0:
            return np.asarray([])
        else:
            return np.zeros((self.numStates,))

    def stateDiffEqs(self, t, y, varGroups, params):
        # Differential equations governing the control states, i.e., their time
        #   derivatives
        if self.numStates == 0:
            return np.asarray([])
        else:
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
        if self.numStates == 0:
            return np.asarray([])
        else:
            return np.zeros((self.numStates, self.numStates))

    def partials_ctrlStateDEQs_wrt_epoch(self, t, y, varGroups, params):
        if self.numStates == 0:
            return np.asarray([])
        else:
            return np.zeros((self.numStates,))

    def partials_ctrlStateDEQs_wrt_params(self, t, y, varGroups, params):
        if self.numStates == 0 or len(params) == 0:
            return np.asarray([])
        else:
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
        return np.asarray([])  # No control states

    def partials_term_wrt_epoch(self, t, y, varGroups, params):
        return np.array([0], ndmin=2)

    def partials_term_wrt_params(self, t, y, varGroups, params):
        partials = np.zeros((1, len(params)))
        partials[0, self.paramIx0] = 1
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
        return np.asarray([])  # No control states

    def partials_term_wrt_epoch(self, t, y, varGroups, params):
        return np.array([0], ndmin=2)

    def partials_term_wrt_params(self, t, y, varGroups, params):
        partials = np.zeros((1, len(params)))
        partials[0, self.paramIx0] = 1
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

    def _getAngles(self, params):
        return params[self.paramIx0], params[self.paramIx0 + 1]

    def evalTerm(self, t, y, varGroups, params):
        alpha, beta = self._getAngles(params)
        return np.asarray(
            [
                [np.cos(beta) * np.cos(alpha)],
                [np.cos(beta) * np.sin(alpha)],
                [np.sin(beta)],
            ]
        )

    def partials_term_wrt_coreState(self, t, y, varGroups, params):
        return np.zeros((3, self.coreStateSize))

    def partials_term_wrt_ctrlState(self, t, y, varGroups, params):
        return np.asarray([])  # no control states

    def partials_term_wrt_epoch(self, t, y, varGroups, params):
        return np.zeros((3, 1))

    def partials_term_wrt_params(self, t, y, varGroups, params):
        partials = np.zeros((3, len(params)))
        alpha, beta = self._getAngles(params)
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

    def _concat(self, *arrays):
        out = arrays[0]
        for array in arrays[1:]:
            if np.asarray(array).size > 0:
                out = np.concatenate((out, array))

        return out

    @property
    def epochIndependent(self):
        return (
            self.force.epochIndependent
            and self.mass.epochIndependent
            and self.orient.epochIndependent
        )

    @property
    def numStates(self):
        return self.force.numStates + self.mass.numStates + self.orient.numStates

    def stateICs(self):
        return self._concat(
            self.force.stateICs, self.mass.stateICs, self.orient.stateICs
        )

    def stateDiffEqs(self, t, y, varGroups, params):
        # Differential equations governing the control states, i.e., their time
        #   derivatives
        return self._concat(
            self.force.stateDiffEqs(t, y, varGroups, params),
            self.mass.stateDiffEqs(t, y, varGroups, params),
            self.orient.stateDiffEqs(t, y, varGroups, params),
        )

    @property
    def stateNames(self):
        # TODO
        pass

    @property
    def params(self):
        return self._concat(self.force.params, self.mass.params, self.orient.params)

    def registerParams(self, ix0):
        # Tells the control terms where their params live within the vector
        self.force.paramIx0 = ix0
        ix0 += len(self.force.params)

        self.mass.paramIx0 = ix0
        ix0 += len(self.mass.params)

        self.orient.paramIx0 = ix0

    def accelVec(self, t, y, varGroups, params):
        # Returns Cartesian acceleration vector
        force = self.force.evalTerm(t, y, varGroups, params)
        mass = self.mass.evalTerm(t, y, varGroups, params)
        vec = self.orient.evalTerm(t, y, varGroups, params)

        return (force / mass) * vec

    def _accelPartials(self, t, y, varGroups, params, partialFcn):
        # Use chain rule to combine partials of the acceleration w.r.t. some other
        #   parameter
        f = self.force.evalTerm(t, y, varGroups, params)
        m = self.mass.evalTerm(t, y, varGroups, params)
        vec = self.orient.evalTerm(t, y, varGroups, params)

        dfdX = getattr(self.force, partialFcn)(t, y, varGroups, params)
        dmdX = getattr(self.mass, partialFcn)(t, y, varGroups, params)
        dodX = getattr(self.orient, partialFcn)(t, y, varGroups, params)

        term1 = (vec @ dfdX / m) if dfdX.size > 0 else 0
        term2 = (-f * vec / (m * m)) @ dmdX if dmdX.size > 0 else 0
        term3 = (f / m) * dodX if dodX.size > 0 else 0

        partials = term1 + term2 + term3
        return np.asarray([]) if isinstance(partials, int) else partials

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
        return self._concat(
            self.force.partials_ctrlStateDEQs_wrt_coreState(t, y, varGroups, params),
            self.mass.partials_ctrlStateDEQs_wrt_coreState(t, y, varGroups, params),
            self.orient.partials_ctrlStateDEQs_wrt_coreState(t, y, varGroups, params),
        )

    def partials_ctrlStateDEQs_wrt_ctrlState(self, t, y, varGroups, params):
        # The partial derivatives of the control state Diff Eqs with respect to
        #   the control state vector
        return self._concat(
            self.force.partials_ctrlStateDEQs_wrt_ctrlState(t, y, varGroups, params),
            self.mass.partials_ctrlStateDEQs_wrt_ctrlState(t, y, varGroups, params),
            self.orient.partials_ctrlStateDEQs_wrt_ctrlState(t, y, varGroups, params),
        )

    def partials_ctrlStateDEQs_wrt_epoch(self, t, y, varGroups, params):
        # The partial derivatives of the control state Diff Eqs with respect to
        #   the epoch
        return self._concat(
            self.force.partials_ctrlStateDEQs_wrt_epoch(t, y, varGroups, params),
            self.mass.partials_ctrlStateDEQs_wrt_epoch(t, y, varGroups, params),
            self.orient.partials_ctrlStateDEQs_wrt_epoch(t, y, varGroups, params),
        )

    def partials_ctrlStateDEQs_wrt_params(self, t, y, varGroups, params):
        # The partial derivatives of the control state Diff Eqs with respect to
        #   the parameters
        return self._concat(
            self.force.partials_ctrlStateDEQs_wrt_params(t, y, varGroups, params),
            self.mass.partials_ctrlStateDEQs_wrt_params(t, y, varGroups, params),
            self.orient.partials_ctrlStateDEQs_wrt_params(t, y, varGroups, params),
        )
