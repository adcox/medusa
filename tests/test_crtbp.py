"""
Test CRTBP dynamics
"""
import numpy as np
import pytest
from conftest import loadBody

from pika.data import Body
from pika.dynamics import EOMType
from pika.dynamics.crtbp import DynamicsModel, ModelConfig

earth = loadBody("Earth")
moon = loadBody("Moon")


@pytest.fixture(scope="module")
def emConfig():
    return ModelConfig(earth, moon)


@pytest.mark.parametrize("bodies", [[earth, moon], [moon, earth]])
def test_modelConfig(bodies):
    config = ModelConfig(*bodies)

    assert "mu" in config.params
    assert config.params["mu"] < 0.5
    assert config.charL > 1.0
    assert config.charM > 1.0
    assert config.charT > 1.0


@pytest.mark.usefixtures("emConfig")
class TestDynamicsModel:
    def test_stateSize(self, emConfig):
        model = DynamicsModel(emConfig)
        assert model.stateSize(EOMType.STATE) == 6
        assert model.stateSize(EOMType.STM) == 36
        assert model.stateSize(EOMType.ALL) == 42
        assert model.stateSize(EOMType.EPOCH_DEPS) == 0
        assert model.stateSize(EOMType.PARAM_DEPS) == 0

    @pytest.mark.parametrize(
        "append", [EOMType.STATE, EOMType.STM, EOMType.EPOCH_DEPS, EOMType.PARAM_DEPS]
    )
    def test_appendICs(self, emConfig, append):
        model = DynamicsModel(emConfig)
        q0 = np.array([0, 1, 2, 3, 4, 5])
        q0_mod = model.appendICs(q0, append)

        assert q0_mod.shape == (q0.size + model.stateSize(append),)
