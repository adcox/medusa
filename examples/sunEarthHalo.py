#!/usr/bin/env python3
import logging
from pathlib import Path

import matplotlib.pyplot as plt
from rich.logging import RichHandler

import medusa.corrections.constraints as constraints
import medusa.plots as plots
from medusa.corrections import (
    ControlPoint,
    DifferentialCorrector,
    LeastSquaresUpdate,
    Segment,
    ShootingProblem,
)
from medusa.crtbp import DynamicsModel
from medusa.data import Body
from medusa.propagate import Propagator

logger = logging.getLogger("medusa")
logger.addHandler(RichHandler(show_time=False, show_path=False, enable_link_path=False))
logger.setLevel(logging.INFO)


BODIES = Path(__file__).parent.parent / "resources/body-data.xml"
assert BODIES.exists(), "Cannot find body-data.xml file"

sun = Body.fromXML(BODIES, "Sun")
earth = Body.fromXML(BODIES, "Earth Barycenter")
model = DynamicsModel(sun, earth)
# print(model.params["mu"])
q0 = [
    9.8881583890062652e-1,
    0.0,
    4.0357157728763445e-4,
    0.0,
    8.8734639842052918e-3,
    0.0,
]
period = 3.06

prop = Propagator(model, dense=False)
solForward = prop.propagate(q0, [0, period / 2], t_eval=[0.0, period / 4, period / 2])
solReverse = prop.propagate(
    q0, [0, -period / 2], t_eval=[0.0, -period / 4, -period / 2]
)
points = [
    ControlPoint.fromProp(solForward, 0),
    ControlPoint.fromProp(solForward, 1),
    ControlPoint.fromProp(solForward, 2),
    ControlPoint.fromProp(solReverse, 1),
    ControlPoint.fromProp(solReverse, 0),
]
segments = [
    Segment(points[ix], period / 4, terminus=points[ix + 1], prop=prop)
    for ix in range(len(points) - 1)
]

# make terminus of final arc the origin of the first
segments[-1].terminus = points[0]

# Use single TOF for all segments
for seg in segments[1:]:
    seg.tof = segments[0].tof

problem = ShootingProblem()
problem.addSegments(segments)

for seg in segments:
    problem.addConstraints(constraints.StateContinuity(seg))

# Constrain z0
problem.addConstraints(
    constraints.VariableValue(
        points[0].state, [None, 0, 4.0357157728763445e-4, None, None, None]
    )
)
problem.build()

problem.printFreeVars()
problem.printConstraints()
problem.printJacobian()

corrector = DifferentialCorrector()
corrector.updateGenerator = LeastSquaresUpdate()
solution, log = corrector.solve(problem)
assert log["status"] == "converged"

fig = plt.figure()
ax = fig.add_subplot(projection="3d")

convert = plots.ToCoordVals(["x", "y", "z"])

it0 = ShootingProblem.fromIteration(solution, log, it=0)
segs0 = convert.segments(it0.segments)
pts0 = convert.controlPoints(it0.controlPoints)

ax.plot(segs0[0], segs0[1], segs0[2], lw=2, label="Init. Guess")
ax.plot(pts0[0], pts0[1], pts0[2], c="k", ls="none", marker=".", ms=8)

segs = convert.segments(solution.segments)
pts = convert.controlPoints(solution.controlPoints)
ax.plot(segs[0], segs[1], segs[2], lw=2, label="Solution")
ax.plot(pts[0], pts[1], pts[2], c="r", ls="none", marker="+", ms=10)

ax.grid()
ax.set_aspect("equal")
ax.legend()

plt.show()
