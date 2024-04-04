"""
Test Propagation
"""
import numpy as np
import pytest
import scipy.optimize
from conftest import loadBody

from pika.dynamics import EOMVars
from pika.dynamics.crtbp import DynamicsModel, ModelConfig
from pika.propagate import propagate


@pytest.fixture
def emModel():
    earth = loadBody("Earth")
    moon = loadBody("Moon")
    return DynamicsModel(ModelConfig(earth, moon))


def test_propagate(emModel):
    # Initial conditions for an Earth-Moon L3 vertical orbit
    y0 = np.array([0.8213, 0.0, 0.5690, 0.0, -1.8214, 0.0])
    tspan = [0, 6.3111]
    sol = propagate(emModel, y0, tspan)

    assert isinstance(sol, scipy.optimize.OptimizeResult)
    assert hasattr(sol, "model")
    assert sol.model == emModel
    assert hasattr(sol, "params")
    assert sol.params == None
    assert hasattr(sol, "eomVars")
    assert sol.eomVars == EOMVars.STATE
