#!/usr/bin/env python3
"""
Test script to plot a ShooterProblem outout
"""
import os
import sys
import logging
from pathlib import Path

import matplotlib.pyplot as plt
from rich.logging import RichHandler

import pika.corrections.constraints as constraints
from pika.corrections import (
    ControlPoint,
    DifferentialCorrector,
    L2NormConvergence,
    MinimumNormUpdate,
    Segment,
    ShootingProblem,
)
from pika.data import Body
from pika.dynamics.crtbp import DynamicsModel
from pika.plots import TrajPlotter
from pika.propagate import Propagator

logger = logging.getLogger("pika")
logger.addHandler(RichHandler(show_time=False, show_path=False, enable_link_path=False))
logger.setLevel(logging.INFO)


BODIES = Path(__file__).parent.parent / "resources/body-data.xml"
assert BODIES.exists(), "Cannot find body-data.xml file"

earth = Body.fromXML(BODIES, "Earth")
moon = Body.fromXML(BODIES, "Moon")
model = DynamicsModel(earth, moon)
q0 = [0.8213, 0.0, 0.5690, 0.0, -1.8214, 0.0]
period = 6.311
prop = Propagator(model, dense=False)
sol = prop.propagate(
    q0,
    [0, period],
    t_eval=[0.0, period / 4, period / 2, 3 * period / 4, period],
)
points = [ControlPoint.fromProp(sol, ix) for ix in range(len(sol.t))]
segments = [
    Segment(points[ix], period / 4, terminus=points[ix + 1], prop=prop)
    for ix in range(len(points) - 1)
]

# make terminus of final arc the origin of the first
segments[-1].terminus = points[0]

problem = ShootingProblem()
problem.addSegments(segments)

# Create continuity constraints
for seg in segments:
    problem.addConstraints(constraints.ContinuityConstraint(seg))

problem.build()
corrector = DifferentialCorrector()
corrector.updateGenerator = MinimumNormUpdate()
corrector.convergenceCheck = L2NormConvergence(1e-8)
solution, log = corrector.solve(problem)

assert isinstance(solution, ShootingProblem)
assert log["status"] == "converged"

plotter = TrajPlotter()
plotter.plot(solution, coords=["x", "y", "z"], plotPrimaries=True)
plt.show()
