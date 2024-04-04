"""
Test CRTBP dynamics
"""
import numpy as np
import pytest
from conftest import loadBody

from pika.data import Body
from pika.dynamics import EOMVars
from pika.dynamics.crtbp import DynamicsModel, ModelConfig

earth = loadBody("Earth")
moon = loadBody("Moon")
sun = loadBody("Sun")


@pytest.fixture(scope="module")
def emConfig():
    return ModelConfig(earth, moon)


class TestModelConfig:
    @pytest.mark.parametrize("bodies", [[earth, moon], [moon, earth]])
    def test_bodyOrder(self, bodies):
        config = ModelConfig(*bodies)

        assert isinstance(config.bodies, tuple)
        assert len(config.bodies) == 2
        assert config.bodies[0].gm > config.bodies[1].gm

        assert "mu" in config.params
        assert config.params["mu"] < 0.5
        assert config.charL > 1.0
        assert config.charM > 1.0
        assert config.charT > 1.0

    def test_equals(self):
        config1 = ModelConfig(earth, moon)
        config2 = ModelConfig(moon, earth)
        config3 = ModelConfig(earth, sun)

        assert config1 == config1
        assert config1 == config2
        assert not config1 == config3


@pytest.mark.usefixtures("emConfig")
class TestDynamicsModel:
    def test_stateSize(self, emConfig):
        model = DynamicsModel(emConfig)
        assert model.stateSize(EOMVars.STATE) == 6
        assert model.stateSize(EOMVars.STM) == 36
        assert model.stateSize([EOMVars.STATE, EOMVars.STM]) == 42
        assert model.stateSize(EOMVars.EPOCH_DEPS) == 0
        assert model.stateSize(EOMVars.PARAM_DEPS) == 0

    @pytest.mark.parametrize(
        "append", [EOMVars.STATE, EOMVars.STM, EOMVars.EPOCH_DEPS, EOMVars.PARAM_DEPS]
    )
    def test_appendICs(self, emConfig, append):
        model = DynamicsModel(emConfig)
        q0 = np.array([0, 1, 2, 3, 4, 5])
        q0_mod = model.appendICs(q0, append)

        assert q0_mod.shape == (q0.size + model.stateSize(append),)
