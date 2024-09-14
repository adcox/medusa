#!/usr/bin/env python3

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rich.logging import RichHandler

logger = logging.getLogger("medusa")
logger.addHandler(RichHandler(show_time=False, show_path=False, enable_link_path=False))
logger.setLevel(logging.INFO)

BODIES = Path(__file__).parent.parent / "resources/body-data.xml"
assert BODIES.exists()


import medusa.corrections as cor
import medusa.corrections.constraints as cons
import medusa.crtbp as cr
import medusa.dynamics.lowthrust as lt
import medusa.plots as plots
from medusa.data import Body
from medusa.propagate import Propagator

earth = Body.fromXML(BODIES, "Earth")
moon = Body.fromXML(BODIES, "Moon")

crtbpModel = cr.DynamicsModel(earth, moon)

# Initial conditions for an L2 lyapunov in the ballistic CR3BP
q0 = [1.11815, 0.0, 0.0, 0.0, 0.1866196, 0.0]
T = 3.42088

# ------------------------------------------------------------------------------
# Get periodic CR3BP orbit
prop = Propagator(crtbpModel, dense=False)
sol = prop.propagate(q0, [0, T], t_eval=[0.0, T / 2, T])
points = [cor.ControlPoint.fromProp(sol, ix) for ix in range(len(sol.t))]
# Make planar
points[1].state.values[1:4] = 0
points[1].state.values[5] = 0
for point in points:
    point.state.mask[1:4] = True  # y0, z0, dx0 are fixed
    point.state.mask[5] = True  # dz0 is fixed

segments = [
    cor.Segment(points[ix], T / 2, terminus=points[ix + 1], prop=prop)
    for ix in range(len(points) - 1)
]

# Make terminus of final arc the origin of the first
segments[-1].tof = segments[0].tof
segments[-1].terminus = points[0]

problem = cor.ShootingProblem()
problem.addSegments(segments)

for seg in segments:
    problem.addConstraints(cons.StateContinuity(seg, indices=[0, 4]))

problem.build()
assert problem.checkJacobian()

corrector = cor.DifferentialCorrector()
solution, log = corrector.solve(problem)

solution.printJacobian()

fig, ax = plt.subplots()

convert = plots.ToCoordVals(["x", "y"])

it0 = cor.ShootingProblem.fromIteration(solution, log, it=0)
sol0 = convert.segments(it0.segments)
solf = convert.segments(solution.segments)

ax.plot(sol0[0], sol0[1], lw=2, label="Init. Guess")
ax.plot(solf[0], solf[1], lw=2, label="CR3BP")

# ------------------------------------------------------------------------------
# Get periodic CR3BP+LT orbit
thrust = lt.control.ConstThrustTerm(7e-3)
mass = lt.control.ConstMassTerm(1.0)
orient = lt.control.ConstOrientTerm(-76.5 * np.pi / 180.0, 0.0)
control = lt.control.ForceMassOrientLaw(thrust, mass, orient)
ltModel = lt.dynamics.LowThrustCrtbpDynamics(earth, moon, control)

# Start from converged CR3BP solution
fv = solution.freeVarVec()
points = [
    cor.ControlPoint(ltModel, 0, [fv[0], 0, 0, 0, fv[1], 0]),
    cor.ControlPoint(ltModel, 0, [fv[2], 0, 0, 0, fv[3], 0]),
]

# Make planar
for point in points:
    point.state.mask[1:3] = True  # y, z are fixed

params = cor.Variable(control.params, mask=True)
segments = [
    cor.Segment(points[ix], fv[4], points[ix + 1 % 2], propParams=params)
    for ix in range(len(points) - 1)
]

ltproblem = cor.ShootingProblem()
ltproblem.addSegments(segments)
for seg in segments:
    ltproblem.addConstraints(cons.StateContinuity(seg))  # , indices=[0,3,4,5]))

ltproblem.build()
# assert ltproblem.checkJacobian(tol=1e-4)
ltsolution, ltlog = corrector.solve(ltproblem)
ltsolution.printJacobian()

sol_lt = convert.segments(ltsolution.segments)
ax.plot(sol_lt[0], sol_lt[1], lw=2, label="Low-Thrust")

ax.grid()
ax.set_aspect("equal")
ax.legend()
plt.show()
