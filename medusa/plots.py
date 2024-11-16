"""
Plotting
========

Plotting tools are defined here.
"""
import numpy as np
import scipy.optimize  # type: ignore

from medusa import util
from medusa.dynamics import State, VarGroup


class ToCoordVals:
    def __init__(self, coords):
        """
        Args:
            coords ([str]): the coordinate name to extract. Can by "t" for the time
                value, or one of the variables names (see
                :func:`medusa.dynamics.DynamicsModel.coords`) from the dynamics
                model.
        """
        # TODO check input types
        self.coords = util.toList(coords)

    def data(self, states):
        """
        Extract time and/or state values via coordinate name

        Args:
            states ([State]): an M-element array of states

        Returns:
            numpy.ndarray: an LxN array of values where L is the number of coordinates
        """
        if isinstance(states, State):
            states = [states]
        allVals = np.asarray(states)
        times = [state.time for state in states]

        # TODO check input types
        # Build a dict mapping coordinate name to state index
        stateNames = states[0].coords(VarGroup.STATE)
        coordMap = {"t": None}
        for ix, name in enumerate(stateNames):
            coordMap[name] = ix

        vals = np.empty((len(self.coords), len(times)))
        for ixv, coord in enumerate(self.coords):
            ixc = coordMap[coord]

            if ixc is None:
                vals[ixv] = times
            else:
                if len(allVals.shape) == 1:
                    vals[ixv] = allVals[ixc]
                if len(allVals.shape) == 2:
                    vals[ixv] = allVals[:, ixc]

        return vals

    def segments(self, segments, insertNaN=True):
        # TODO check input types
        # TODO accept varGroups as input to class? or to method?
        vals = None
        for seg in segments:
            _vals = self.propagation(seg.propagate(VarGroup.STATE))
            if vals is None:
                vals = _vals
            else:
                # Concatenate data with a set of NaN in between so that breaks
                #   between segments are not connected
                if insertNaN:
                    brk = np.full((_vals.shape[0], 1), np.nan)
                    vals = np.concatenate((vals, brk, _vals), axis=1)
                else:
                    vals = np.concatenate((vals, _vals), axis=1)
        return vals

    def controlPoints(self, points):
        # TODO check input types
        vals = None
        for point in points:
            _vals = self.data(point.state)
            if vals is None:
                vals = _vals
            else:
                vals = np.concatenate((vals, _vals), axis=1)

        return vals

    def propagation(self, result):
        if not isinstance(result, scipy.optimize.OptimizeResult):
            raise TypeError("Input must by a propagation solution")

        # TODO allow different time sampling if propSol.sol is not None?
        return self.data(result.states)
