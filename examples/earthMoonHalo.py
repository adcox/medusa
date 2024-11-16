#!/usr/bin/env python3
import logging
from pathlib import Path

from rich.logging import RichHandler

import medusa.corrections.constraints as constraints
from medusa.corrections import (
    ControlPoint,
    DifferentialCorrector,
    Segment,
    ShootingProblem,
)
from medusa.data import Body
from medusa.dynamics import VarGroup
from medusa.dynamics.crtbp import DynamicsModel, State
from medusa.propagate import Propagator

logger = logging.getLogger("medusa")
logger.addHandler(RichHandler(show_time=False, show_path=False, enable_link_path=False))
logger.setLevel(logging.DEBUG)


BODIES = Path(__file__).parent.parent / "resources/body-data.xml"
assert BODIES.exists(), "Cannot find body-data.xml file"

earth = Body.fromXML(BODIES, "Earth")
moon = Body.fromXML(BODIES, "Moon")
model = DynamicsModel(earth, moon)
q0 = State(
    model,
    [
        6.4260881453410956e-1,
        -5.9743133135791852e-27,
        7.5004250912552883e-1,
        -1.7560161626620319e-12,
        3.5068017486608094e-1,
        2.6279755586942682e-12,
    ],
)
q0.fillDefaultICs(VarGroup.STM)
period = 3.00

model.checkPartials(q0, [0, period])

prop = Propagator(dense_output=False)
sol = prop.propagate(q0, [0, period], t_eval=[0.0, period / 2, period])
points = [ControlPoint.fromProp(sol, ix) for ix in range(len(sol.t))]
segments = [
    Segment(points[ix], period / 2, terminus=points[ix + 1], prop=prop)
    for ix in range(len(points) - 1)
]

# make terminus of final arc the origin of the first
segments[-1].tof = segments[0].tof
segments[-1].terminus = points[0]

problem = ShootingProblem()
problem.addSegments(segments)

for seg in segments:
    problem.addConstraints(constraints.StateContinuity(seg))

problem.build()

problem.printFreeVars()
problem.printConstraints()
problem.printJacobian()
problem.checkJacobian()

corrector = DifferentialCorrector()
solution, log = corrector.solve(problem)

solution.printJacobian()

assert log["status"] == "converged"

import matplotlib.pyplot as plt

import medusa.plots as plots

for seg in solution.segments:
    seg.denseEval()

convert = plots.ToCoordVals(["x", "y", "z"])
segs = convert.segments(solution.segments)
pts = convert.controlPoints(solution.controlPoints)

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot(segs[0], segs[1], segs[2], lw=2)
ax.plot(pts[0], pts[1], pts[2], c="k", ls="none", marker=".", ms=8)
ax.grid()
ax.set_aspect("equal")

plt.show()
