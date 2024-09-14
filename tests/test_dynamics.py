"""
Test basic dynamics
"""
import numpy as np
import pytest
from conftest import loadBody

from medusa.dynamics import AbstractDynamicsModel, VarGroup

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
            2 * (VarGroup.STATE in varGroups)
            + 4 * (VarGroup.STM in varGroups)
            + 2 * (VarGroup.EPOCH_PARTIALS in varGroups)
            + 6 * (VarGroup.PARAM_PARTIALS in varGroups)
        )


class TestAbstractDynamicsModel:
    @pytest.fixture(scope="class")
    def model(self):
        return DummyModel(sun, earth, moon)

    @pytest.mark.parametrize("bodies, properties", [
        [ [earth], {} ],
        [ [earth, moon], {"b": 21} ],
        [ [earth, moon, sun], {} ],
    ])
    def test_constructor(self, bodies, properties):
        DummyModel(*bodies, **properties)

    @pytest.mark.parametrize("bodies", [
        [earth, "Moon"],
        [301, 302],
    ])
    def test_constructorErr(self, bodies):
        with pytest.raises(TypeError):
            DummyModel(*bodies)

    @pytest.mark.parametrize(
        "varGroups, sz",
        [
            [VarGroup.STATE, 2],
            [VarGroup.STM, 4],
            [VarGroup.EPOCH_PARTIALS, 2],
            [VarGroup.PARAM_PARTIALS, 6],
            [[VarGroup.STATE, VarGroup.STM], 6],
            [[VarGroup.STATE, VarGroup.STM, VarGroup.EPOCH_PARTIALS], 8],
            [
                [
                    VarGroup.STATE,
                    VarGroup.STM,
                    VarGroup.EPOCH_PARTIALS,
                    VarGroup.PARAM_PARTIALS,
                ],
                14,
            ],
        ],
    )
    def test_stateSize(self, model, varGroups, sz):
        assert model.stateSize(varGroups) == sz

    @pytest.mark.parametrize("varIn", [None, [0, 1, 2, 3]])
    @pytest.mark.parametrize(
        "varGroup, shape, out",
        [
            [VarGroup.STATE, (2,), [0, 1]],
            [VarGroup.STM, (2, 2), [[2, 3], [4, 5]]],
            [VarGroup.EPOCH_PARTIALS, (2,), [6, 7]],
            [VarGroup.PARAM_PARTIALS, (2, 3), [[8, 9, 10], [11, 12, 13]]],
        ],
    )
    def test_extractVars(self, model, varGroup, shape, out, varIn):
        # standard use case: y has all the variable groups, we want subset out
        y = np.arange(14)
        varOut = model.extractVars(y, varGroup, varGroupsIn=varIn)
        assert isinstance(varOut, np.ndarray)
        assert varOut.shape == shape
        np.testing.assert_array_equal(varOut, out)

    @pytest.mark.parametrize(
        "y, varIn, varOut, yOut",
        [
            [
                np.arange(4),
                [VarGroup.STATE, VarGroup.EPOCH_PARTIALS],
                VarGroup.EPOCH_PARTIALS,
                [2, 3],
            ],
            [
                np.arange(8),
                [VarGroup.STATE, VarGroup.PARAM_PARTIALS],
                VarGroup.PARAM_PARTIALS,
                [[2, 3, 4], [5, 6, 7]],
            ],
        ],
    )
    def test_extractVars_notFull(self, model, y, varIn, varOut, yOut):
        # Test extracting from a vector that doesn't include all the VarGroup
        assert np.array_equal(model.extractVars(y, varOut, varIn), yOut)

    @pytest.mark.parametrize(
        "y, varIn, varOut",
        [
            [np.arange(2), VarGroup.STATE, VarGroup.STM],
            [np.arange(2), VarGroup.STATE, VarGroup.EPOCH_PARTIALS],
            [np.arange(2), VarGroup.STATE, VarGroup.PARAM_PARTIALS],
            [np.arange(4), [VarGroup.STATE, VarGroup.EPOCH_PARTIALS], VarGroup.STM],
            [
                np.arange(4),
                [VarGroup.STATE, VarGroup.EPOCH_PARTIALS],
                VarGroup.PARAM_PARTIALS,
            ],
            [np.arange(8), [VarGroup.STATE, VarGroup.PARAM_PARTIALS], VarGroup.STM],
            [
                np.arange(8),
                [VarGroup.STATE, VarGroup.PARAM_PARTIALS],
                VarGroup.EPOCH_PARTIALS,
            ],
        ],
    )
    def test_extractVars_missingVarIn(self, model, y, varIn, varOut):
        # varIn doesn't include the desired varOut
        with pytest.raises(RuntimeError):
            model.extractVars(y, varOut, varIn)

    @pytest.mark.parametrize(
        "y, varIn, varOut",
        [
            [np.arange(3), [VarGroup.STATE, VarGroup.STM], VarGroup.STM]
        ]
    )
    def test_extractVars_missingData(self, model, y, varIn, varOut):
        with pytest.raises(ValueError):
            model.extractVars(y, varOut, varIn)

    @pytest.mark.parametrize(
        "VarGroup, out",
        [
            [VarGroup.STATE, [0] * 2],
            [VarGroup.STM, [1, 0, 0, 1]],
            [VarGroup.EPOCH_PARTIALS, [0] * 2],
            [VarGroup.PARAM_PARTIALS, [0] * 6],
        ],
    )
    def test_defaultICs(self, model, VarGroup, out):
        assert model.defaultICs(VarGroup).tolist() == out

    @pytest.mark.parametrize(
        "VarGroup, appended",
        [
            [VarGroup.STM, [1, 0, 0, 1]],
            [VarGroup.EPOCH_PARTIALS, [0] * 2],
            [VarGroup.PARAM_PARTIALS, [0] * 6],
            [VarGroup.STATE, [0] * 2],
            [[VarGroup.STM, VarGroup.EPOCH_PARTIALS], [1, 0, 0, 1, 0, 0]],
            [[VarGroup.EPOCH_PARTIALS, VarGroup.STM], [1, 0, 0, 1, 0, 0]],
        ],
    )
    def test_appendICs(self, model, VarGroup, appended):
        y = np.arange(3)  # arbitrary vector
        y0 = model.appendICs(y, VarGroup)
        assert np.array_equal(y0[:3], y)
        assert y0[3:].tolist() == appended

    @pytest.mark.parametrize(
        "varGroups, tf",
        [
            [VarGroup.STATE, True],
            [VarGroup.STM, False],
            [VarGroup.EPOCH_PARTIALS, False],
            [VarGroup.PARAM_PARTIALS, False],
            [[VarGroup.STATE, VarGroup.STM], True],
        ],
    )
    def test_validForPropagation(self, model, varGroups, tf):
        assert model.validForPropagation(varGroups) == tf

    def test_varNames(self, model):
        stateNames = model.varNames(VarGroup.STATE)
        assert stateNames == ["State 0", "State 1"], stateNames

        stmNames = model.varNames(VarGroup.STM)
        assert stmNames == ["STM(0,0)", "STM(0,1)", "STM(1,0)", "STM(1,1)"], stmNames

        epochNames = model.varNames(VarGroup.EPOCH_PARTIALS)
        assert epochNames == ["Epoch Dep 0", "Epoch Dep 1"], epochNames

        paramNames = model.varNames(VarGroup.PARAM_PARTIALS)
        assert paramNames == [
            "Param Dep(0,0)",
            "Param Dep(0,1)",
            "Param Dep(0,2)",
            "Param Dep(1,0)",
            "Param Dep(1,1)",
            "Param Dep(1,2)",
        ], paramNames

    @pytest.mark.parametrize("grp", [7, -1])
    def test_varNames_invalid(self, model, grp):
        with pytest.raises(ValueError):
            model.varNames(grp)

    def test_eq(self):
        model = DummyModel(earth, sun)
        model2 = DummyModel(earth, sun)

        assert model == model2
        assert not model == "abc"
