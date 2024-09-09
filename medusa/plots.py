"""
Plotting
"""
import numpy as np
import scipy.optimize

from medusa import util
from medusa.dynamics import VarGroups


class ToCoordVals:
    def __init__(self, coords):
        """
        Args:
            coords ([str]): the coordinate name to extract. Can by "t" for the time
                value, or one of the variables names (see
                :func:`medusa.dynamics.DynamicsModel.varNames`) from the dynamics
                model.
        """
        # TODO check input types
        self.coords = util.toList(coords)

    def data(self, model, times, states):
        """
        Extract time and/or state values via coordinate name

        Args:
            model (medusa.dynamics.DynamicsModel): the model associated with the
                ``states``
            times ([float], numpy.ndarray<float>): a 1D array of N times
            states ([[float]], numpy.ndarray<flat>): an MxN array of states

        Returns:
            numpy.ndarray: an LxN array of values where L is the number of coordinates
        """
        # TODO check input types
        # Build a dict mapping coordinate name to state index
        stateNames = model.varNames(VarGroups.STATE)
        coordMap = {"t": None}
        for ix, name in enumerate(stateNames):
            coordMap[name] = ix

        vals = np.empty((len(self.coords), len(times)))
        for ixv, coord in enumerate(self.coords):
            ixc = coordMap[coord]

            if ixc is None:
                vals[ixv] = times
            else:
                if len(states.shape) == 1:
                    vals[ixv] = states[ixc]
                if len(states.shape) == 2:
                    vals[ixv] = states[ixc, :]

        return vals

    def segments(self, segments, insertNaN=True):
        # TODO check input types
        # TODO accept varGroups as input to class? or to method?
        vals = None
        for seg in segments:
            _vals = self.propagation(seg.propagate(VarGroups.STATE))
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
            _vals = self.data(
                point.model, [point.epoch.allVals[0]], point.state.allVals
            )
            if vals is None:
                vals = _vals
            else:
                vals = np.concatenate((vals, _vals), axis=1)

        return vals

    def propagation(self, result):
        if not isinstance(result, scipy.optimize.OptimizeResult):
            raise TypeError("Input must by a propagation solution")

        # TODO allow different time sampling if propSol.sol is not None?
        return self.data(result.model, result.t, result.y)


# def plotPrimaries(ax, model, coords):
#    # TODO what is t? what are params?
#    t = 0
#    params = []
#    if model is None:
#        raise RuntimeError("Could not find a dynamics model")
#
#    handles = []
#    for ix, body in enumerate(model.bodies):
#        state = model.bodyState(ix, t, params)
#        vals = getCoordVals(model, t, state, coords)
#        handles.append(ax.plot(*vals, "k.", markersize=16))
#
#    return handles

# def plotTraj(obj, coords=["x", "y"], primaries=False, fig=None):
#    pltKwargs = {}
#    if len(coords) == 1:
#        coords = ["t", coords[0]]
#    elif len(coords) == 3:
#        pltKwargs["projection"] = "3d"
#    elif len(coords) > 3:
#        raise IndexError("Cannot plot more than three coordinates")
#
#    if fig is None:
#        fig = plt.figure()
#        ax = fig.add_subplot(**pltKwargs)
#    else:
#        ax = fig.axes[0]
#
#    model = None
#
#    if isinstance(obj, scipy.optimize.OptimizeResult):
#        plotPropagation(ax, obj, coords)
#        model = obj.model
#    elif isinstance(obj, corrections.ControlPoint):
#        plotControlPoint(ax, obj, coords)
#        model = obj.model
#    elif isinstance(obj, corrections.Segment):
#        plotSegment(ax, obj, coords)
#        model = obj.origin.model
#    elif isinstance(obj, corrections.ShootingProblem):
#        model = obj._segments[0].origin.model
#        for seg in obj._segments:
#            plotSegment(ax, seg, coords)
#
#    if primaries:
#        plotPrimaries(ax, model, coords)
#
#    ax.grid()
#    ax.set_xlabel(coords[0])
#    ax.set_ylabel(coords[1])
#
#    if len(coords) < 3:
#        ax.set_aspect(1)
#    else:
#        ax.set_aspect("equal")
#        ax.set_zlabel(coords[2])
#
#    return fig
