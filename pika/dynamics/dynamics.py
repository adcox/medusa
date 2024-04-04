"""
Dynamics Classes and Interfaces
"""
from abc import ABC, abstractmethod

import numpy as np


class SystemData(ABC):
    def __init__(bodies):
        self.bodies = copy(bodies)
        self.params = {}

    @property
    @abstractmethod
    def charL(self):
        pass

    @property
    @abstractmethod
    def charT(self):
        pass

    @property
    @abstractmethod
    def charM(self):
        pass


class IDynamicsModel(ABC):
    @property
    @abstractmethod
    def systemData(self):
        pass

    @abstractmethod
    def equationsOfMotion(self, eqType, params):
        pass

    @abstractmethod
    def bodyPos(self, ix, t):
        pass

    @abstractmethod
    def bodyVel(self, ix, t):
        pass

    @abstractmethod
    def stateSize(self, eqType):
        pass

    @property
    @abstractmethod
    def numParams(self):
        pass
