"""
Plotting
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

from pika import corrections

# TODO this should be defined in each dynamics model
coordMap = {
    "t": None,
    "x": 0,
    "y": 1,
    "z": 2,
    "dx": 3,
    "dy": 4,
    "dz": 5,
}


class TrajPlotter:
    def __init__(self):
        self._objects = {}
        self.plotPrimaries = False

    def plot(self, obj, coords=["x", "y"], fig=None, **kwargs):
        if fig is None:
            pltKwargs = {}
            if len(coords) == 1:
                coords = ["t", coords[0]]
            elif len(coords) == 3:
                pltKwargs["projection"] = "3d"
            elif len(coords) > 3:
                raise IndexError("Cannot plot more than three coordinates")
            fig = plt.figure()
            ax = fig.add_subplot(**pltKwargs)
        else:
            raise NotImplementedError("TODO: support plotting on existing figure")

        model = None

        if isinstance(obj, scipy.optimize.OptimizeResult):
            self._plotPropagation(ax, obj, coords)
            model = obj.model
        elif isinstance(obj, corrections.ControlPoint):
            self._plotControlPoint(ax, obj, coords)
            model = obj.model
        elif isinstance(obj, corrections.Segment):
            self._plotSegment(ax, obj, coords)
            model = obj.origin.model
        elif isinstance(obj, corrections.ShootingProblem):
            model = obj._segments[0].origin.model
            for seg in obj._segments:
                self._plotSegment(ax, seg, coords)

        if kwargs.get("plotPrimaries", self.plotPrimaries):
            # TODO what is t? what are params?
            t = 0
            params = []
            if model is None:
                raise RuntimeError("Could not find a dynamics model")
            for ix, body in enumerate(model.bodies):
                pos = model.bodyPos(ix, t, params)
                state = np.append(pos, model.bodyVel(ix, t, params))
                vals = self._getVals(t, state, coords)
                ax.plot(*vals, "k.", markersize=16)

        ax.grid()
        ax.set_xlabel(coords[0])
        ax.set_ylabel(coords[1])

        if len(coords) < 3:
            ax.set_aspect(1)
        else:
            ax.set_aspect("equal")
            ax.set_zlabel(coords[2])

        return fig

    def _getVals(self, times, states, coords):
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

    def _plotSegment(self, ax, segment, coords, **kwargs):
        if segment.propSol is None:
            segment.propagate(EOMVars.STATE)

        self._plotPropagation(ax, segment.propSol, coords, **kwargs)

        if not segment.origin in self._objects:
            self._plotControlPoint(ax, segment.origin, coords)

        if segment.terminus is not None and not segment.terminus in self._objects:
            self._plotControlPoint(ax, segment.terminus, coords)

    def _plotControlPoint(self, ax, point, coords, **kwargs):
        vals = self._getVals(point.epoch.allVals[0], point.state.allVals, coords)
        marker = kwargs.get("marker", ".")
        ms = kwargs.get("markersize", 10)
        col = kwargs.get("color", "gray")
        self._objects[point] = ax.plot(*vals, c=col, marker=marker, markersize=ms)

    def _plotPropagation(self, ax, propSol, coords, **kwargs):
        if getattr(propSol, "sol", None) is None:
            vals = self._getVals(propSol.t, propSol.y, coords)
            ax.plot(*vals, **kwargs)
        else:
            # TODO choose a sampling frequency and evaluate propSol.sol
            raise NotImplementedError()
