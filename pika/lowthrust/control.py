"""
Low-thrust control objects
"""
from abc import ABC, abstractmethod

import numpy as np

# ------------------------------------------------------------------------------
# Control Terms
# ------------------------------------------------------------------------------


class ControlTerm(ABC):
    """
    Represents a term in the control equations

    .. note:: This is an abstract class and cannot be instantiated.

    Attributes:
        epochIndependent (bool): whether or not this term is independent of the epoch
        numStates (int): the number of extra state variables this term defines
        params (list): the default parameter values
        paramIx0 (int): the index of the first parameter "owned" by this term
            within the full parameter list.
        stateICs (numpy.ndarray): the initial conditions for the state variables
            this term defines.
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
        """
        Defines the differential equations that govern the state variables this
        term defines, i.e., derivatives of the state variables with respect to
        integration time.

        The input arguments are consistent with those passed to the
        :func:`pika.dynamics.AbstractDynamicsModel.diffEqs` function.

        .. note:: This method is implemented to return zeros for all state variables
           by default. Override it to define custom behavior.

        Returns:
            numpy.ndarray: the time derivatives of the state variables. If this
            term doesn't define any state variables, an empty array is returned.
        """
        if self.numStates == 0:
            return np.asarray([])
        else:
            return np.zeros((self.numStates,))

    @abstractmethod
    def evalTerm(self, t, y, varGroups, params):
        """
        Evaluate the term.

        The input arguments are consistent with those passed to the
        :func:`pika.dynamics.AbstractDynamicsModel.diffEqs` function.

        Returns:
            float, numpy.ndarray: the evaluated term
        """
        pass

    @abstractmethod
    def partials_term_wrt_coreState(self, t, y, varGroups, params):
        """
        Compute the partial derivatives of :func:`evalTerm` with respect to the
        "core state", i.e., the state variables that exist independently of the
        control parametrization.

        The input arguments are consistent with those passed to the
        :func:`pika.dynamics.AbstractDynamicsModel.diffEqs` function.

        Returns:
            numpy.ndarray: the partial derivatives where the rows represent the
            elements in :func:`evalTerm` and the columns represent the core states.
        """
        pass

    @abstractmethod
    def partials_term_wrt_ctrlState(self, t, y, varGroups, params):
        """
        Compute the partial derivatives of :func:`evalTerm` with respect to the
        control state variables that are defined by *this term*.

        The input arguments are consistent with those passed to the
        :func:`pika.dynamics.AbstractDynamicsModel.diffEqs` function.

        Returns:
            numpy.ndarray: the partial derivatives
        """
        pass

    @abstractmethod
    def partials_term_wrt_epoch(self, t, y, varGroups, params):
        """
        Compute the partial derivatives of :func:`evalTerm` with respect to the epoch.

        The input arguments are consistent with those passed to the
        :func:`pika.dynamics.AbstractDynamicsModel.diffEqs` function.

        Returns:
            numpy.ndarray: the partial derivatives where the rows represent the
            elements in :func:`evalTerm` and the column represents the epoch.
        """
        pass

    @abstractmethod
    def partials_term_wrt_params(self, t, y, varGroups, params):
        """
        Compute the partial derivatives of :func:`evalTerm` with respect to the
        parameters *this term* defines.

        The input arguments are consistent with those passed to the
        :func:`pika.dynamics.AbstractDynamicsModel.diffEqs` function.

        Returns:
            numpy.ndarray: the partial derivatives where the rows represent the
            elements in :func:`evalTerm` and the columns represent the parameters.
        """
        pass

    def partials_coreStateDEQs_wrt_ctrlState(self, t, y, varGroups, params):
        """
        Compute the partial derivatives of the core state differential equations
        (defined in :func:`~pika.dynamics.AbstractDynamicsModel.diffEqs`) with
        respect to the control state variables that are defined by this term.

        The input arguments are consistent with those passed to the
        :func:`pika.dynamics.AbstractDynamicsModel.diffEqs` function.

        Returns:
            numpy.ndarray: the partial derivatives where the rows represent the
            core states and the columns represent the control states defined by
            this term. If this term doesn't define any control states, an
            empty array is returned.
        """
        if self.numStates == 0:
            return np.asarray([])
        else:
            return np.zeros((self.coreStateSize, self.numStates))

    def partials_ctrlStateDEQs_wrt_coreState(self, t, y, varGroups, params):
        """
        Compute the partial derivatives of :func:`stateDiffEqs` with respect to
        the "core state," i.e., the state variables that exist independent of the
        control parameterization.

        The input arguments are consistent with those passed to the
        :func:`pika.dynamics.AbstractDynamicsModel.diffEqs` function.

        Returns:
            numpy.ndarray: the partial derivatives where the rows represent the
            elements in :func:`stateDiffEqs` and the columns represent the core
            states. If this term doesn't define any control states, an empty
            array is returned.
        """
        if self.numStates == 0:
            return np.asarray([])
        else:
            return np.zeros((self.numStates, self.coreStateSize))

    def partials_ctrlStateDEQs_wrt_ctrlState(self, t, y, varGroups, params):
        """
        Compute the partial derivatives of :func:`stateDiffEqs` with respect to
        the control state variables defined by this term.

        The input arguments are consistent with those passed to the
        :func:`pika.dynamics.AbstractDynamicsModel.diffEqs` function.

        Returns:
            numpy.ndarray: the partial derivatives where the rows represent the
            elements in :func:`stateDiffEqs` and the columns represent the core
            states. If this term doesn't define any control states, an empty
            array is returned.
        """
        if self.numStates == 0:
            return np.asarray([])
        else:
            return np.zeros((self.numStates, self.numStates))

    def partials_ctrlStateDEQs_wrt_epoch(self, t, y, varGroups, params):
        """
        Compute the partial derivatives of :func:`stateDiffEqs` with respect to
        the epoch.

        The input arguments are consistent with those passed to the
        :func:`pika.dynamics.AbstractDynamicsModel.diffEqs` function.

        Returns:
            numpy.ndarray: the partial derivatives where the rows represent the
            elements in :func:`stateDiffEqs` and the column represents epoch.
            If this term doesn't define any control states, an empty array is
            returned.
        """
        if self.numStates == 0:
            return np.asarray([])
        else:
            return np.zeros((self.numStates,))

    def partials_ctrlStateDEQs_wrt_params(self, t, y, varGroups, params):
        """
        Compute the partial derivatives of :func:`stateDiffEqs` with respect to
        the parameters defined by this term.

        The input arguments are consistent with those passed to the
        :func:`pika.dynamics.AbstractDynamicsModel.diffEqs` function.

        Returns:
            numpy.ndarray: the partial derivatives where the rows represent the
            elements in :func:`stateDiffEqs` and the columns represent the
            parameters. If this term doesn't define any control states, an empty
            array is returned.
        """
        if self.numStates == 0 or len(params) == 0:
            return np.asarray([])
        else:
            return np.zeros((self.numStates, len(params)))


class ConstThrustTerm(ControlTerm):
    """
    Defines a constant thrust. The thrust is stored as a parameter.

    Args:
        thrust (float): the thrust force in units consistent with the model
            (i.e., if the model nondimensionalizes values, this value should
            also be nondimensionalized).
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
    Defines a constant mass. The mass is stored as a parameter.

    Args:
        mass (float): the mass in units consistent with the model
            (i.e., if the model nondimensionalizes values, this value should
            also be nondimensionalized).
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

    Args:
        alpha (float): the angle between the projection of the thrust vector
            into the xy-plane and the x-axis, measured about the z-axis. Units
            are radians.
        beta (float): the angle between the thrust vector and the xy-plane. A
            positive value corresponds to a positive z-component. Units are
            radians.
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


# ------------------------------------------------------------------------------
# Control Laws
# ------------------------------------------------------------------------------


class ControlLaw(ABC):
    """
    Interface definition for a low-thrust control law

    Attributes:
        epochIndepdnent (bool): whether or not the control parameterization is
            epoch-independent.
        numStates (int): the number of state variables defined by the control law
        stateNames (list of str): the names of the state variables defined by
            the control law
        params (list of float): the default parameter values for the control law
    """

    @abstractmethod
    def accelVec(self, t, y, varGroups, params):
        """
        Compute the acceleration vector delivered by this control law.

        The input arguments are consistent with those passed to the
        :func:`pika.dynamics.AbstractDynamicsModel.diffEqs` function.

        Returns:
            numpy.ndarray: a 3x1 array that gives the Cartesian acceleration
            vector.
        """
        pass

    @property
    @abstractmethod
    def epochIndependent(self):
        pass

    @property
    @abstractmethod
    def numStates(self):
        pass

    @property
    @abstractmethod
    def stateNames(self):
        pass

    @property
    @abstractmethod
    def params(self):
        pass

    @abstractmethod
    def stateDiffEqs(self, t, y, varGroups, params):
        """
        Defines the differential equations that govern the state variables this
        control law defines, i.e., derivatives of the state variables with respect
        to integration time.

        The input arguments are consistent with those passed to the
        :func:`pika.dynamics.AbstractDynamicsModel.diffEqs` function.

        Returns:
            numpy.ndarray: the time derivatives of the state variables. If this
            term doesn't define any state variables, an empty array is returned.
        """
        pass

    @abstractmethod
    def partials_accel_wrt_coreState(self, t, y, varGroups, params):
        """
        The partial derivatives of :func:`accelVec` with respect to the "core state,"
        i.e., the state variables that exist independent of the control
        parameterization.

        The input arguments are consistent with those passed to the
        :func:`pika.dynamics.AbstractDynamicsModel.diffEqs` function.

        Returns:
            numpy.ndarray: the partial derivatives; the rows represent the elements
            of the acceleration vector and the columns represent the core state
            variables.
        """
        pass

    @abstractmethod
    def partials_accel_wrt_ctrlState(self, t, y, varGroups, params):
        """
        The partial derivatives of :func:`accelVec` with respect to the control
        states defined by the control law.

        The input arguments are consistent with those passed to the
        :func:`pika.dynamics.AbstractDynamicsModel.diffEqs` function.

        Returns:
            numpy.ndarray: the partial derivatives; the rows represent the elements
            of the acceleration vector and the columns represent the control state
            variables.
        """
        pass

    @abstractmethod
    def partials_accel_wrt_epoch(self, t, y, varGroups, params):
        """
        The partial derivatives of :func:`accelVec` with respect to the epoch.

        The input arguments are consistent with those passed to the
        :func:`pika.dynamics.AbstractDynamicsModel.diffEqs` function.

        Returns:
            numpy.ndarray: the partial derivatives; the rows represent the elements
            of the acceleration vector and the column represents the epoch.
        """
        pass

    @abstractmethod
    def partials_accel_wrt_params(self, t, y, varGroups, params):
        """
        The partial derivatives of :func:`accelVec` with respect to the parameters
        defined by the control law.

        The input arguments are consistent with those passed to the
        :func:`pika.dynamics.AbstractDynamicsModel.diffEqs` function.

        Returns:
            numpy.ndarray: the partial derivatives; the rows represent the elements
            of the acceleration vector and the columns represent the parameters.
        """
        pass

    @abstractmethod
    def partials_ctrlStateDEQs_wrt_coreState(self, t, y, varGroups, params):
        """
        The partial derivatives of :func:`stateDiffEqs` with respect to the "core
        states," i.e., the state variables that exist independently of the
        control parameterization.

        The input arguments are consistent with those passed to the
        :func:`pika.dynamics.AbstractDynamicsModel.diffEqs` function.

        Returns:
            numpy.ndarray: the partial derivatives; the rows represent the
            differential equations and the columns represent the core state
            variables.
        """
        pass

    @abstractmethod
    def partials_ctrlStateDEQs_wrt_ctrlState(self, t, y, varGroups, params):
        """
        The partial derivatives of :func:`stateDiffEqs` with respect to the state
        variables defined by the control law.

        The input arguments are consistent with those passed to the
        :func:`pika.dynamics.AbstractDynamicsModel.diffEqs` function.

        Returns:
            numpy.ndarray: the partial derivatives; the rows represent the
            differential equations and the columns represent the control state
            variables.
        """
        pass

    @abstractmethod
    def partials_ctrlStateDEQs_wrt_epoch(self, t, y, varGroups, params):
        """
        The partial derivatives of :func:`stateDiffEqs` with respect to the epoch.

        The input arguments are consistent with those passed to the
        :func:`pika.dynamics.AbstractDynamicsModel.diffEqs` function.

        Returns:
            numpy.ndarray: the partial derivatives; the rows represent the
            differential equations and the column represents the epoch.
        """
        pass

    @abstractmethod
    def partials_ctrlStateDEQs_wrt_params(self, t, y, varGroups, params):
        """
        The partial derivatives of :func:`stateDiffEqs` with respect to the
        parameters defined by the control law.

        The input arguments are consistent with those passed to the
        :func:`pika.dynamics.AbstractDynamicsModel.diffEqs` function.

        Returns:
            numpy.ndarray: the partial derivatives; the rows represent the
            differential equations and the columns represent the parameters.
        """
        pass


class SeparableControlLaw(ControlLaw):
    """
    A control law implementation with "separable" terms. In this context,
    separable means that each term defines state variables and parameters
    independently of the other terms. As a result, the control state
    differential equations and their partial derivatives can be concatenated
    without additional calculus to relate the terms.

    .. note:: This implementation is abstract; derived objects must define the
       acceleration vector and its associated partial derivatives.

    Args:
        terms (ControlTerm): the term(s) to include in the control parameterization.
    """

    def __init__(self, *terms):
        self.terms = tuple(terms)

    def _concat(self, arrays):
        """
        Concatenate numpy arrays. This convenience method skips concatenation of
        empty arrays, avoiding errors.
        """
        out = arrays[0]
        for array in arrays[1:]:
            if np.asarray(array).size > 0:
                out = np.concatenate((out, array))

        return out

    @property
    def epochIndependent(self):
        return all(term.epochIndependent for term in self.terms)

    @property
    def numStates(self):
        return sum(term.numStates for term in self.terms)

    @property
    def stateNames(self):
        # TODO
        pass

    @property
    def params(self):
        return self._concat([term.params for term in self.terms])

    def stateICs(self):
        return self._concat([term.stateICs for term in self.terms])

    def stateDiffEqs(self, t, y, varGroups, params):
        return self._concat(
            [term.stateDiffEqs(t, y, varGroups, params) for term in self.terms]
        )

    def registerParams(self, ix0):
        """
        Set the ``paramIx0`` attribute of the control terms

        Args:
            ix0 (int): the index of the first control parameter within the full
                set of parameters.
        """
        for term in self.terms:
            term.paramIx0 = ix0
            ix0 += len(term.params)

    def partials_ctrlStateDEQs_wrt_coreState(self, t, y, varGroups, params):
        # Because the control terms are independent, we can just concatenate
        #   the partial derivatives of the control state diff eqs.
        return self._concat(
            [
                term.partials_ctrlStateDEQs_wrt_coreState(t, y, varGroups, params)
                for term in self.terms
            ]
        )

    def partials_ctrlStateDEQs_wrt_ctrlState(self, t, y, varGroups, params):
        return self._concat(
            [
                term.partials_ctrlStateDEQs_wrt_ctrlState(t, y, varGroups, params)
                for term in self.terms
            ]
        )

    def partials_ctrlStateDEQs_wrt_epoch(self, t, y, varGroups, params):
        return self._concat(
            [
                term.partials_ctrlStateDEQs_wrt_epoch(t, y, varGroups, params)
                for term in self.terms
            ]
        )

    def partials_ctrlStateDEQs_wrt_params(self, t, y, varGroups, params):
        return self._concat(
            [
                term.partials_ctrlStateDEQs_wrt_params(t, y, varGroups, params)
                for term in self.terms
            ]
        )


class ForceMassOrientLaw(SeparableControlLaw):
    """
    A separable control law that accepts three terms: force, mass, and orientation.

    Args:
        force (ControlTerm): defines the scalar thrust force
        mass (ControlTerm): defines the scalar mass
        orient (ControlTerm): defines the unit vector that orients the thrust

    The acceleration is computed as: :math:`\\vec{a} = \\frac{f}{m} \\hat{u}`
    where :math:`f` is the thrust force, :math:`m` is the mass, and
    :math:`\\hat{u}` is the orientation.
    """

    def __init__(self, force, mass, orient):
        super().__init__(force, mass, orient)

    def accelVec(self, t, y, varGroups, params):
        # Returns Cartesian acceleration vector
        force = self.terms[0].evalTerm(t, y, varGroups, params)
        mass = self.terms[1].evalTerm(t, y, varGroups, params)
        vec = self.terms[2].evalTerm(t, y, varGroups, params)

        return (force / mass) * vec

    def _accelPartials(self, t, y, varGroups, params, partialFcn):
        # Use chain rule to combine partials of the acceleration w.r.t. some other
        #   parameter
        f = self.terms[0].evalTerm(t, y, varGroups, params)
        m = self.terms[1].evalTerm(t, y, varGroups, params)
        vec = self.terms[2].evalTerm(t, y, varGroups, params)

        dfdX = getattr(self.terms[0], partialFcn)(t, y, varGroups, params)
        dmdX = getattr(self.terms[1], partialFcn)(t, y, varGroups, params)
        dodX = getattr(self.terms[2], partialFcn)(t, y, varGroups, params)

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
