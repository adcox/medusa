"""
Test basic dynamics
"""
import numpy as np
import pytest
from conftest import loadBody

from medusa.dynamics import AbstractDynamicsModel, VarGroups

earth, moon, sun = loadBody("Earth"), loadBody("Moon"), loadBody("Sun")


class DummyModel(AbstractDynamicsModel):
    """
    A dummy model to test base class functionality with
    """

    def bodyState(self, ix, t, params):
        pass

    def diffEqs(self, t, y, varGroups, params):
        pass

    def epochIndependent(self):
        return False

    def stateSize(self, varGroups):
        # A model with 2 state variables, epoch dependencies, and three parameters
        varGroups = np.array(varGroups, ndmin=1)
        return (
            2 * (VarGroups.STATE in varGroups)
            + 4 * (VarGroups.STM in varGroups)
            + 2 * (VarGroups.EPOCH_PARTIALS in varGroups)
            + 6 * (VarGroups.PARAM_PARTIALS in varGroups)
        )


class TestAbstractDynamicsModel:
    @pytest.fixture(scope="class")
    def model(self):
        return DummyModel(sun, earth, moon)

    @pytest.mark.parametrize(
        "varGroups, sz",
        [
            [VarGroups.STATE, 2],
            [VarGroups.STM, 4],
            [VarGroups.EPOCH_PARTIALS, 2],
            [VarGroups.PARAM_PARTIALS, 6],
            [[VarGroups.STATE, VarGroups.STM], 6],
            [[VarGroups.STATE, VarGroups.STM, VarGroups.EPOCH_PARTIALS], 8],
            [
                [
                    VarGroups.STATE,
                    VarGroups.STM,
                    VarGroups.EPOCH_PARTIALS,
                    VarGroups.PARAM_PARTIALS,
                ],
                14,
            ],
        ],
    )
    def test_stateSize(self, model, varGroups, sz):
        assert model.stateSize(varGroups) == sz

    @pytest.mark.parametrize("varIn", [None, [0, 1, 2, 3]])
    @pytest.mark.parametrize(
        "varGroups, shape, out",
        [
            [VarGroups.STATE, (2,), [0, 1]],
            [VarGroups.STM, (2, 2), [[2, 3], [4, 5]]],
            [VarGroups.EPOCH_PARTIALS, (2,), [6, 7]],
            [VarGroups.PARAM_PARTIALS, (2, 3), [[8, 9, 10], [11, 12, 13]]],
        ],
    )
    def test_extractVars(self, model, varGroups, shape, out, varIn):
        # standard use case: y has all the variable groups, we want subset out
        y = np.arange(14)
        varOut = model.extractVars(y, varGroups, varGroupsIn=varIn)
        assert isinstance(varOut, np.ndarray)
        assert varOut.shape == shape
        np.testing.assert_array_equal(varOut, out)

    @pytest.mark.parametrize(
        "y, varIn, varOut, yOut",
        [
            [
                np.arange(4),
                [VarGroups.STATE, VarGroups.EPOCH_PARTIALS],
                VarGroups.EPOCH_PARTIALS,
                [2, 3],
            ],
            [
                np.arange(8),
                [VarGroups.STATE, VarGroups.PARAM_PARTIALS],
                VarGroups.PARAM_PARTIALS,
                [[2, 3, 4], [5, 6, 7]],
            ],
        ],
    )
    def test_extractVars_notFull(self, model, y, varIn, varOut, yOut):
        assert np.array_equal(model.extractVars(y, varOut, varIn), yOut)

    @pytest.mark.parametrize(
        "y, varIn, varOut",
        [
            [np.arange(2), VarGroups.STATE, VarGroups.STM],
            [np.arange(2), VarGroups.STATE, VarGroups.EPOCH_PARTIALS],
            [np.arange(2), VarGroups.STATE, VarGroups.PARAM_PARTIALS],
            [np.arange(4), [VarGroups.STATE, VarGroups.EPOCH_PARTIALS], VarGroups.STM],
            [
                np.arange(4),
                [VarGroups.STATE, VarGroups.EPOCH_PARTIALS],
                VarGroups.PARAM_PARTIALS,
            ],
            [np.arange(8), [VarGroups.STATE, VarGroups.PARAM_PARTIALS], VarGroups.STM],
            [
                np.arange(8),
                [VarGroups.STATE, VarGroups.PARAM_PARTIALS],
                VarGroups.EPOCH_PARTIALS,
            ],
        ],
    )
    def test_extractVars_invalid(self, model, y, varIn, varOut):
        with pytest.raises(RuntimeError):
            model.extractVars(y, varOut, varIn)

    @pytest.mark.parametrize(
        "varGroups, out",
        [
            [VarGroups.STATE, [0] * 2],
            [VarGroups.STM, [1, 0, 0, 1]],
            [VarGroups.EPOCH_PARTIALS, [0] * 2],
            [VarGroups.PARAM_PARTIALS, [0] * 6],
        ],
    )
    def test_defaultICs(self, model, varGroups, out):
        assert model.defaultICs(varGroups).tolist() == out

    @pytest.mark.parametrize(
        "varGroups, appended",
        [
            [VarGroups.STM, [1, 0, 0, 1]],
            [VarGroups.EPOCH_PARTIALS, [0] * 2],
            [VarGroups.PARAM_PARTIALS, [0] * 6],
            [VarGroups.STATE, [0] * 2],
            [[VarGroups.STM, VarGroups.EPOCH_PARTIALS], [1, 0, 0, 1, 0, 0]],
            [[VarGroups.EPOCH_PARTIALS, VarGroups.STM], [1, 0, 0, 1, 0, 0]],
        ],
    )
    def test_appendICs(self, model, varGroups, appended):
        y = np.arange(3)  # arbitrary vector
        y0 = model.appendICs(y, varGroups)
        assert np.array_equal(y0[:3], y)
        assert y0[3:].tolist() == appended

    @pytest.mark.parametrize(
        "varGroups, tf",
        [
            [VarGroups.STATE, True],
            [VarGroups.STM, False],
            [VarGroups.EPOCH_PARTIALS, False],
            [VarGroups.PARAM_PARTIALS, False],
            [[VarGroups.STATE, VarGroups.STM], True],
        ],
    )
    def test_validForPropagation(self, model, varGroups, tf):
        assert model.validForPropagation(varGroups) == tf

    def test_varNames(self, model):
        stateNames = model.varNames(VarGroups.STATE)
        assert stateNames == ["State 0", "State 1"], stateNames

        stmNames = model.varNames(VarGroups.STM)
        assert stmNames == ["STM(0,0)", "STM(0,1)", "STM(1,0)", "STM(1,1)"], stmNames

        epochNames = model.varNames(VarGroups.EPOCH_PARTIALS)
        assert epochNames == ["Epoch Dep 0", "Epoch Dep 1"], epochNames

        paramNames = model.varNames(VarGroups.PARAM_PARTIALS)
        assert paramNames == [
            "Param Dep(0,0)",
            "Param Dep(0,1)",
            "Param Dep(0,2)",
            "Param Dep(1,0)",
            "Param Dep(1,1)",
            "Param Dep(1,2)",
        ], paramNames

    # TODO test __eq__
