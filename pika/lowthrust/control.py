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
        self.coreStateSize = 6  # TODO
        self.paramIx0 = None

    @property
    def epochIndependent(self):
        return True

    @property
    def params(self):
        return []  # TODO return None?

    @property
    def numStates(self):
        return 0

    @property
    def stateICs(self):
        return np.zeros((self.numCtrlStates,))

    def stateDiffEqs(self, t, y, varGroups, params):
        # Differential equations governing the control states, i.e., their time
        #   derivatives
        return np.zeros((self.numCtrlStates,))

    @abstractmethod
    def evalTerm(self, t, y, varGroups, params):
        # Evaluate the control term and return it
        pass

    @abstractmethod
    def partials_term_wrt_time(self, t, y, varGroups, params):
        pass

    @abstractmethod
    def partials_term_wrt_coreState(self, t, y, varGroups, params):
        pass

    @abstractmethod
    def partials_term_wrt_epoch(self, t, y, varGroups, params):
        pass

    @abstractmethod
    def partials_term_wrt_params(self, t, y, varGroups, params):
        pass

    @abstractmethod
    def partials_coreStateDEQs_wrt_ctrlState(self, t, y, varGroups, params):
        # The partial derivatives of the core STATE Diff Eqs with
        #   respect to the control state vector
        # TODO I don't think this is necessary; write out derivation to be sure
        pass

    def partials_ctrlStateDEQs_wrt_coreState(self, t, y, varGroups, params):
        # The partial derivatives of the control state Diff Eqs with respect to
        #   the core state vector
        return np.zeros((self.numCtrlStates, self.coreStateSize))

    def partials_ctrlStateDEQs_wrt_ctrlState(self, t, y, varGroups, params):
        # The partial derivatives of the control state Diff Eqs with respect to
        #   the control state vector
        return np.zeros((self.numCtrlStates, self.numCtrlStates))

    def partials_ctrlState_wrt_epoch(self, t, y, varGroups, params):
        return np.zeros((self.numCtrlStates,))

    def partials_ctrlState_wrt_params(self, t, y, varGroups, params):
        return np.zeros((self.numCtrlStates, len(params)))


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

    def partials_term_wrt_time(self, t, y, varGroups, params):
        return np.asarray([0])

    def partials_term_wrt_coreState(self, t, y, varGroups, params):
        return np.asarray([0])

    def partials_term_wrt_epoch(self, t, y, varGroups, params):
        return np.asarray([0])

    def partials_term_wrt_params(self, t, y, varGroups, params):
        partials = np.zeros(1, len(params))
        partials[self.paramIx0] = 1
        return partials


class ControlLaw:
    """
    A control law that augments the natural acceleration acting on the body
    """

    def __init__(self, forcePolicy, massPolicy, orientPolicy):
        self.forcePolicy = forcePolicy
        self.massPolicy = massPolicy
        self.orientPolicy = orientPolicy

    @property
    def numStates(self):
        return (
            self.forcePolicy.numCore
            + self.massPolicy.numCore
            + self.orientPolicy.numCore
        )

    def stateICs(self):
        # TODO
        pass

    def stateDiffEqs(self, t, y, varGroups, params):
        # Differential equations governing the control states, i.e., their time
        #   derivatives
        # TODO
        pass

    @property
    def stateNames(self):
        # TODO
        pass

    @property
    def params(self):
        return (
            self.forcePolicy.params + self.massPolicy.params + self.orientPolicy.params
        )

    def registerParams(self, ix0):
        # Tells the control terms where their params live within the vector
        self.forcePolicy.paramIx0 = ix0
        ix0 += len(self.forcePolicy.params)

        self.massPolicy.paramIx0 = ix0
        ix0 += len(self.massPolicy.params)

        self.orientPolicy.paramIx0 = ix0

    def accelVec(t, y, varGroups, params):
        # Returns Cartesian acceleration vector
        force = self.forcePolicy.eval(t, y, varGroups, params)
        mass = self.massPolicty.eval(t, y, varGroups, params)
        vec = self.orientPolicy.eval(t, y, varGroups, params)

        return (force / mass) * vec

    def partials_accel_wrt_coreState(self, t, y, varGroups, params):
        force = self.forcePolicy.eval(t, y, varGroups, params)
        mass = self.massPolicty.eval(t, y, varGroups, params)
        vec = self.orientPolicy.eval(t, y, varGroups, params)

        dfdt = self.forcePolicy.partials_term_wrt_coreState(t, y, varGroups, params)
        dmdt = self.massPolicy.partials_term_wrt_coreState(t, y, varGroups, params)
        dodt = self.orientPolicy.partials_term_wrt_coreState(t, y, varGroups, params)

        return (
            dfdt * vec / mass + force * np.log(mass) * vec * dmdt + force * dodt / mass
        )

    def partials_accel_wrt_ctrlState(self, t, y, varGroups, params):
        pass  # TODO

    def partials_accel_wrt_epoch(self, t, y, varGroups, params):
        force = self.forcePolicy.eval(t, y, varGroups, params)
        mass = self.massPolicty.eval(t, y, varGroups, params)
        vec = self.orientPolicy.eval(t, y, varGroups, params)

        dfdT = self.forcePolicy.partials_term_wrt_epoch(t, y, varGroups, params)
        dmdT = self.massPolicy.partials_term_wrt_epoch(t, y, varGroups, params)
        dodT = self.orientPolicy.partials_term_wrt_epoch(t, y, varGroups, params)

        return (
            dfdT * vec / mass + force * np.log(mass) * vec * dmdT + force * dodT / mass
        )

    def partials_accel_wrt_params(self, t, y, varGroups, params):
        force = self.forcePolicy.eval(t, y, varGroups, params)
        mass = self.massPolicty.eval(t, y, varGroups, params)
        vec = self.orientPolicy.eval(t, y, varGroups, params)

        dfdp = self.forcePolicy.partials_term_wrt_params(t, y, varGroups, params)
        dmdp = self.massPolicy.partials_term_wrt_params(t, y, varGroups, params)
        dodp = self.orientPolicy.partials_term_wrt_params(t, y, varGroups, params)

        # TODO adjust for matrix notation
        return (
            dfdp * vec / mass + force * np.log(mass) * vec * dmdp + force * dodp / mass
        )

    def partials_ctrlStateDEQs_wrt_coreState(self, t, y, varGroups, params):
        # The partial derivatives of the control state Diff Eqs with respect to
        #   the core state vector
        pass

    def partials_ctrlStateDEQs_wrt_ctrlState(self, t, y, varGroups, params):
        # The partial derivatives of the control state Diff Eqs with respect to
        #   the control state vector
        pass

    def partials_ctrlStateDEQs_wrt_epoch(self, t, y, varGroups, params):
        # The partial derivatives of the control state Diff Eqs with respect to
        #   the epoch
        pass

    def partials_ctrlStateDEQs_wrt_params(self, t, y, varGroups, params):
        # The partial derivatives of the control state Diff Eqs with respect to
        #   the parameters
        pass
