"""
Dynamics Classes and Interfaces
"""
from abc import ABC, abstractmethod
from enum import Flag, auto

import numpy as np

from pika.data import Body


class EOMVars(Flag):
    """
    Specify the variables included in the equations of motion
    """

    STATE = auto()
    """: State variables; usually includes the position and velocity vectors """

    STM = auto()
    """: State Transition Matrix; N**2 set of state variable partials """

    EPOCH_DEPS = auto()
    """: Epoch dependencies """

    PARAM_DEPS = auto()
    """: Parameter dependencies """

    ALL = STATE | STM | EPOCH_DEPS | PARAM_DEPS
    """: All variables """


class ModelConfig:
    """
    Basic class to store a model configuration

    Attributes:
        bodies (tuple): a tuple of :class:`~pika.data.Body` objects
        params (dict): the parameters associated with this configuration
    """

    def __init__(self, *bodies, **params):
        if any([not isinstance(body, Body) for body in bodies]):
            raise TypeError("Expecting Body objects")

        # Copy body objects into tuple
        self.bodies = (copy(body) for body in bodies)

        # Unpack parameters into internal dict
        self._params = {**params}

        # Define default characteristic quantities as unity
        self._charL = 1.0  # km
        self._charT = 1.0  # sec
        self._charM = 1.0  # kg

    @property
    def params(self):
        return self._params.copy()

    @property
    def charL(self):
        return self._charL

    @property
    def charT(self):
        return self._charT

    @property
    def charM(self):
        return self._charM


class AbstractDynamicsModel(ABC):
    """
    Contains the mathematics that define a dynamical model

    Attributes:
        config (ModelConfig): model configuration
        numParams (int): number of parameters this model supports
    """

    def __init__(self, config):
        if not isinstance(config, ModelConfig):
            raise TypeError("config input must be a ModelConfig object")

        self.config = config

    @abstractmethod
    def evalEOMs(self, t, q, eomVars):
        pass

    @abstractmethod
    def bodyPos(self, ix, t):
        pass

    @abstractmethod
    def bodyVel(self, ix, t):
        pass

    @abstractmethod
    def stateSize(self, eomVars):
        pass

    def defaultICs(self, eomVars):
        """
        Get the default initial conditions for a set of equations. This basic
        implementation returns a flattened identity matrix for the :attr:`~EOMVars.STM`
        and zeros for the other equation types. Derived classes can override
        this method to provide other values.

        Args:
            eomVars (EOMVars): equation type

        Returns:
            numpy.ndarray: initial conditions for the specified equation type
        """
        if eomVars == EOMVars.ALL:
            raise NotImplementedError("Does not support generation of ALL EOMVars")
        elif eomVars == EOMVars.STM:
            return np.identity(self.stateSize(EOMVars.STATE)).flatten()
        else:
            return np.zeros((self.stateSize(eomVars),))

    def appendICs(self, q, varsToAppend):
        """
        Append initial conditions for the specified variable groups to the
        provided state vector

        Args:
            q (numpy.ndarray): state vector of arbitrary length
            varsToAppend (EOMVars): the variable group(s) to append initial
                conditions for. Note that

        Returns:
            numpy.ndarray: an initial condition vector, duplicating ``q`` at
            the start of the array with the additional initial conditions
            appended afterward
        """
        nIn = q.size
        nOut = self.stateSize(varsToAppend)
        qOut = np.zeros((nIn + nOut,))
        qOut[:nIn] = q
        ix = nIn
        for attr in ["STATE", "STM", "EPOCH_DEPS", "PARAM_DEPS"]:
            tp = getattr(EOMVars, attr)
            if tp in varsToAppend:
                ic = self.defaultICs(tp)
                if ic.size > 0:
                    qOut[ix : ix + ic.size] = ic
                    ix += ic.size

        return qOut

    def validForPropagation(self, eomVars):
        """
        Check that the set of EOM variables can be propagated.

        In many cases, some of the variables are dependent upon others. E.g.,
        the STM equations of motion generally require the state variables to be
        propagated alongside the STM so ``EOMVars.STM`` would be an invalid set for
        evaluation but ``EOMVars.STATE | EOMVars.STM`` would be valid.

        Args:
            eomVars (EOMVars): the variables to be propagated

        Returns:
            bool: True if the set is valid, False otherwise
        """
        # General principle: STATE vars are always required
        return EOMVars.STATE in eomVars
