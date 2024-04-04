"""
Dynamics Classes and Interfaces
"""
from abc import ABC, abstractmethod
from copy import copy
from enum import IntEnum

import numba
import numpy as np

from pika.data import Body


# numba only supports Enum and IntEnum
class EOMVars(IntEnum):
    """
    Specify the variables included in the equations of motion
    """

    STATE = 0
    """: State variables; usually includes the position and velocity vectors """

    STM = 1
    """: State Transition Matrix; N**2 set of state variable partials """

    EPOCH_DEPS = 2
    """: Epoch dependencies """

    PARAM_DEPS = 3
    """: Parameter dependencies """


class ModelConfig:
    """
    Basic class to store a model configuration

    Attributes:
        bodies (tuple): a tuple of :class:`~pika.data.Body` objects
        params (dict): the parameters associated with this configuration
        charL (float):
        charT (float):
        charM (float):
    """

    def __init__(self, *bodies, **params):
        if any([not isinstance(body, Body) for body in bodies]):
            raise TypeError("Expecting Body objects")

        # Copy body objects into tuple
        self.bodies = tuple(copy(body) for body in bodies)

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

    def __eq__(self, other):
        """
        Compare two ModelConfig objects. This can be overridden for more specific
        comparisons in derived classes
        """
        if not isinstance(other, ModelConfig):
            return False

        if not type(self) == type(other):
            return False

        if not all([b1 == b2 for b1, b2 in zip(self.bodies, other.bodies)]):
            return False

        return (
            self.params == other.params
            and self.charL == other.charL
            and self.charT == other.charT
            and self.charM == other.charM
        )


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
    def evalEOMs(self, t, y, params, eomVars):
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
        if eomVars == EOMVars.STM:
            return np.identity(self.stateSize(EOMVars.STATE)).flatten()
        else:
            return np.zeros((self.stateSize(eomVars),))

    def appendICs(self, y0, varsToAppend):
        """
        Append initial conditions for the specified variable groups to the
        provided state vector

        Args:
            y0 (numpy.ndarray): state vector of arbitrary length
            varsToAppend (EOMVars): the variable group(s) to append initial
                conditions for. Note that

        Returns:
            numpy.ndarray: an initial condition vector, duplicating ``q`` at
            the start of the array with the additional initial conditions
            appended afterward
        """
        varsToAppend = np.array(varsToAppend, ndmin=1)
        nIn = y0.size
        nOut = self.stateSize(varsToAppend)
        y0_out = np.zeros((nIn + nOut,))
        y0_out[:nIn] = y0
        ix = nIn
        for v in varsToAppend:
            ic = self.defaultICs(v)
            if ic.size > 0:
                y0_out[ix : ix + ic.size] = ic
                ix += ic.size

        return y0_out

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
        return EOMVars.STATE in np.array(eomVars, ndmin=1)

    def __eq__(self, other):
        if not isinstance(other, AbstractDynamicsModel):
            return False

        return type(self) == type(other) and self.config == other.config
