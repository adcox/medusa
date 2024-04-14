"""
Test basic dynamics
"""
import numpy as np
import pytest
from conftest import loadBody

from pika.dynamics import AbstractDynamicsModel, EOMVars, ModelConfig

earth, moon, sun = loadBody("Earth"), loadBody("Moon"), loadBody("Sun")


class DummyModel(AbstractDynamicsModel):
    """
    A dummy model to test base class functionality with
    """

    def bodyPos(self, ix, t, params):
        pass

    def bodyVel(self, ix, t, params):
        pass

    def evalEOMs(self, t, y, eomVars, params):
        pass

    def epochIndependent(self):
        return False

    def stateSize(self, eomVars):
        # A model with 2 state variables, epoch dependencies, and three parameters
        eomVars = np.array(eomVars, ndmin=1)
        return (
            2 * (EOMVars.STATE in eomVars)
            + 4 * (EOMVars.STM in eomVars)
            + 2 * (EOMVars.EPOCH_DEPS in eomVars)
            + 6 * (EOMVars.PARAM_DEPS in eomVars)
        )


class TestModelConfig:
    # TODO test
    pass


class TestAbstractDynamicsModel:
    @pytest.fixture(scope="class")
    def model(self):
        config = ModelConfig(sun, earth, moon)
        return DummyModel(config)

    @pytest.mark.parametrize(
        "eomVars, sz",
        [
            [EOMVars.STATE, 2],
            [EOMVars.STM, 4],
            [EOMVars.EPOCH_DEPS, 2],
            [EOMVars.PARAM_DEPS, 6],
            [[EOMVars.STATE, EOMVars.STM], 6],
            [[EOMVars.STATE, EOMVars.STM, EOMVars.EPOCH_DEPS], 8],
            [[EOMVars.STATE, EOMVars.STM, EOMVars.EPOCH_DEPS, EOMVars.PARAM_DEPS], 14],
        ],
    )
    def test_stateSize(self, model, eomVars, sz):
        assert model.stateSize(eomVars) == sz

    @pytest.mark.parametrize(
        "eomVars, out",
        [
            [EOMVars.STATE, [0, 1]],
            [EOMVars.STM, [[2, 3], [4, 5]]],
            [EOMVars.EPOCH_DEPS, [6, 7]],
            [EOMVars.PARAM_DEPS, [[8, 9], [10, 11], [12, 13]]],
        ],
    )
    def test_extractVars(self, model, eomVars, out):
        # standard use case: y has all the variable groups, we want subset out
        y = np.arange(14)
        assert model.extractVars(y, eomVars).tolist() == out

    @pytest.mark.parametrize(
        "y, varIn, varOut, yOut",
        [
            [
                np.arange(4),
                [EOMVars.STATE, EOMVars.EPOCH_DEPS],
                EOMVars.EPOCH_DEPS,
                [2, 3],
            ],
            [
                np.arange(8),
                [EOMVars.STATE, EOMVars.PARAM_DEPS],
                EOMVars.PARAM_DEPS,
                [[2, 3], [4, 5], [6, 7]],
            ],
        ],
    )
    def test_extractVars_notFull(self, model, y, varIn, varOut, yOut):
        assert np.array_equal(model.extractVars(y, varOut, varIn), yOut)

    @pytest.mark.parametrize(
        "y, varIn, varOut",
        [
            [np.arange(2), EOMVars.STATE, EOMVars.STM],
            [np.arange(2), EOMVars.STATE, EOMVars.EPOCH_DEPS],
            [np.arange(2), EOMVars.STATE, EOMVars.PARAM_DEPS],
            [np.arange(4), [EOMVars.STATE, EOMVars.EPOCH_DEPS], EOMVars.STM],
            [np.arange(4), [EOMVars.STATE, EOMVars.EPOCH_DEPS], EOMVars.PARAM_DEPS],
            [np.arange(8), [EOMVars.STATE, EOMVars.PARAM_DEPS], EOMVars.STM],
            [np.arange(8), [EOMVars.STATE, EOMVars.PARAM_DEPS], EOMVars.EPOCH_DEPS],
        ],
    )
    def test_extractVars_invalid(self, model, y, varIn, varOut):
        with pytest.raises(RuntimeError):
            model.extractVars(y, varOut, varIn)

    @pytest.mark.parametrize(
        "eomVars, out",
        [
            [EOMVars.STATE, [0] * 2],
            [EOMVars.STM, [1, 0, 0, 1]],
            [EOMVars.EPOCH_DEPS, [0] * 2],
            [EOMVars.PARAM_DEPS, [0] * 6],
        ],
    )
    def test_defaultICs(self, model, eomVars, out):
        assert model.defaultICs(eomVars).tolist() == out

    @pytest.mark.parametrize(
        "eomVars, appended",
        [
            [EOMVars.STM, [1, 0, 0, 1]],
            [EOMVars.EPOCH_DEPS, [0] * 2],
            [EOMVars.PARAM_DEPS, [0] * 6],
            [EOMVars.STATE, [0] * 2],
            [[EOMVars.STM, EOMVars.EPOCH_DEPS], [1, 0, 0, 1, 0, 0]],
            [[EOMVars.EPOCH_DEPS, EOMVars.STM], [1, 0, 0, 1, 0, 0]],
        ],
    )
    def test_appendICs(self, model, eomVars, appended):
        y = np.arange(3)  # arbitrary vector
        y0 = model.appendICs(y, eomVars)
        assert np.array_equal(y0[:3], y)
        assert y0[3:].tolist() == appended

    @pytest.mark.parametrize(
        "eomVars, tf",
        [
            [EOMVars.STATE, True],
            [EOMVars.STM, False],
            [EOMVars.EPOCH_DEPS, False],
            [EOMVars.PARAM_DEPS, False],
            [[EOMVars.STATE, EOMVars.STM], True],
        ],
    )
    def test_validForPropagation(self, model, eomVars, tf):
        assert model.validForPropagation(eomVars) == tf

    # TODO test __eq__
