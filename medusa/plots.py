"""
Plotting
"""
from copy import deepcopy

import matplotlib.pyplot as plt
import scipy.optimize

from medusa import corrections
from medusa.dynamics import VarGroups


def _getVals(model, times, states, coords):
    # Build a dict mapping coordinate name to state index
    stateNames = model.varNames(VarGroups.STATE)
    coordMap = {"t": None}
    for ix, name in enumerate(stateNames):
        coordMap[name] = ix

    vals = []
    for coord in coords:
        ix = coordMap[coord]
        if ix is None:
            vals.append(times)
        else:
            if len(states.shape) == 1:
                vals.append(states[ix])
            if len(states.shape) == 2:
                vals.append(states[ix, :])

    return vals


def objToArray(obj, coords=["x", "y"]):
    # Init
    model, times, states = None, None, None

    if isinstance(obj, corrections.ShootingProblem):
        vals = []  # TODO what is the right concatenation strategy?
        for seg in obj._segments:
            vals.append(objToArray(seg))
        return vals

    if isinstance(obj, corrections.Segment):
        obj = obj.propSol

    if isinstance(obj, scipy.optimize.OptimizeResult):
        # TODO allow different time sampling if propSol.sol is not None?
        model = obj.model
        times = obj.t
        states = obj.y
    elif isinstance(obj, corrections.ControlPoint):
        model = obj.model
        times = obj.epoch.allVals[0]
        states = obj.state.allVals

    if any(x is None for x in [model, times, states]):
        raise NotImplementedError(f"Could not convert {obj} to array data")

    return _getVals(model, times, states, coords)


# def plotSegment(ax, segment, coords, **kwargs):
#    if segment.propSol is None:
#        segment.propagate(VarGroups.STATE)
#
#    return (
#        plotPropagation(ax, segment.propSol, coords, **kwargs),
#        plotControlPoint(ax, segment.origin, coords, **kwargs),
#        plotControlPoint(ax, segment.terminus, coords, **kwargs),
#    )
#
#
# def plotControlPoint(ax, point, coords, **kwargs):
#    marker = kwargs.get("marker", ".")
#    ms = kwargs.get("markersize", 10)
#    col = kwargs.get("color", "gray")
#
#    vals = _getVals(point.model, point.epoch.allVals[0], point.state.allVals, coords)
#    return ax.plot(*vals, c=col, marker=marker, markersize=ms)


def plotPrimaries(ax, model, coords):
    # TODO what is t? what are params?
    t = 0
    params = []
    if model is None:
        raise RuntimeError("Could not find a dynamics model")

    handles = []
    for ix, body in enumerate(model.bodies):
        state = model.bodyState(ix, t, params)
        vals = _getVals(model, t, state, coords)
        handles.append(ax.plot(*vals, "k.", markersize=16))

    return handles


# def plotPropagation(ax, propSol, coords, **kwargs):
#    if getattr(propSol, "sol", None) is None:
#        vals = _getVals(propSol.model, propSol.t, propSol.y, coords)
#        return ax.plot(*vals, **kwargs)
#    else:
#        # TODO choose a sampling frequency and evaluate propSol.sol
#        raise NotImplementedError()


def plotTraj(obj, coords=["x", "y"], primaries=False, fig=None):
    pltKwargs = {}
    if len(coords) == 1:
        coords = ["t", coords[0]]
    elif len(coords) == 3:
        pltKwargs["projection"] = "3d"
    elif len(coords) > 3:
        raise IndexError("Cannot plot more than three coordinates")

    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(**pltKwargs)
    else:
        ax = fig.axes[0]

    model = None

    if isinstance(obj, scipy.optimize.OptimizeResult):
        plotPropagation(ax, obj, coords)
        model = obj.model
    elif isinstance(obj, corrections.ControlPoint):
        plotControlPoint(ax, obj, coords)
        model = obj.model
    elif isinstance(obj, corrections.Segment):
        plotSegment(ax, obj, coords)
        model = obj.origin.model
    elif isinstance(obj, corrections.ShootingProblem):
        model = obj._segments[0].origin.model
        for seg in obj._segments:
            plotSegment(ax, seg, coords)

    if primaries:
        plotPrimaries(ax, model, coords)

    ax.grid()
    ax.set_xlabel(coords[0])
    ax.set_ylabel(coords[1])

    if len(coords) < 3:
        ax.set_aspect(1)
    else:
        ax.set_aspect("equal")
        ax.set_zlabel(coords[2])

    return fig


def plotIteration(
    problem, correctorLog, coords=["x", "y"], it=-1, primaries=False, fig=None
):
    for ix in it:
        freevars = correctorLog["iterations"][ix]["free-vars"]
        prob = deepcopy(problem)
        prob.updateFreeVars(freevars)
        fig = plotTraj(prob, coords=coords, primaries=primaries, fig=fig)

    return fig
