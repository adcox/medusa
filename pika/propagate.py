"""
Propagation, i.e., Numerical Integration
"""
from copy import copy

from scipy.integrate import solve_ivp

from pika.dynamics import EOMVars

DEFAULT_PROP_ARGS = {
    "method": "DOP853",
    "dense_output": True,
    "aTol": 1e-14,
    "rTol": 1e-11,
}


def propagate(model, y0, tSpan, *, params=None, eomVars=EOMVars.STATE, **kwargs):
    # Process inputs
    kwargs_in = {**DEFAULT_PROP_ARGS, **kwargs}
    # TODO warn if overwrite
    kwargs_in["args"] = (params, eomVars)

    # Run propagation
    sol = solve_ivp(model.evalEOMs, tSpan, y0, **kwargs_in)

    # Append our own metadata
    sol.model = copy(model)
    sol.params = copy(params)
    sol.eomVars = eomVars

    return sol
