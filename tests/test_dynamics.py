"""
Test basic dynamics
"""
from copy import copy

import numpy as np
import pint
import pytest
from conftest import loadBody

from medusa.dynamics import AbstractDynamicsModel, VarGroup
from medusa.units import LU, MU, TU, UU, kg, km, sec

earth, moon, sun = loadBody("Earth"), loadBody("Moon"), loadBody("Sun")


class DummyModel(AbstractDynamicsModel):
    """
    A dummy model to test base class functionality with
    """

    def __init__(self, bodies, charL=2 * km, charT=3 * sec, charM=4 * kg):
        super().__init__(bodies, charL, charT, charM)

    def params(self):
        return []

    def bodyState(self, ix, t, params):
        pass

    def diffEqs(self, t, y, varGroups, params):
        pass

    def epochIndependent(self):
        return False

    def groupSize(self, varGroups):
        # A model with 2 state variables, epoch dependencies, and three parameters
        varGroups = np.array(varGroups, ndmin=1)
        return (
            2 * (VarGroup.STATE in varGroups)
            + 4 * (VarGroup.STM in varGroups)
            + 2 * (VarGroup.EPOCH_PARTIALS in varGroups)
            + 6 * (VarGroup.PARAM_PARTIALS in varGroups)
        )

    def varUnits(self, varGroup):
        if varGroup == VarGroup.STATE:
            # let's say the state is a position and a velocity
            return [LU, LU / TU]
        elif varGroup == VarGroup.STM:
            # STM units are [1, time, 1/time, 1]
            return [UU, TU, 1 / TU, UU]
        elif varGroup == VarGroup.EPOCH_PARTIALS:
            return [1 / TU, 1 / TU]  # completely fictional
        elif varGroup == VarGroup.PARAM_PARTIALS:
            return [MU, MU / TU, MU * LU / TU**2] * 2  # completely fictional


class TestAbstractDynamicsModel:
    @pytest.fixture(scope="class")
    def model(self):
        return DummyModel([sun, earth, moon])

    @pytest.mark.parametrize(
        "bodies, charL, charT, charM",
        [
            [[earth], None, None, None],
            [[earth], 3 * km, 45 * sec, 125 * kg],
            [[earth, moon], None, None, None],
            [[earth, moon, sun], None, None, None],
        ],
    )
    def test_constructor(self, bodies, charL, charT, charM):
        quant = {}
        if charL is not None:
            quant["charL"] = charL
        if charT is not None:
            quant["charT"] = charT
        if charM is not None:
            quant["charM"] = charM

        preReg = AbstractDynamicsModel._registry.copy()
        model = DummyModel(bodies, **quant)
        assert len(AbstractDynamicsModel._registry) - len(preReg) == 1

        for b in bodies:
            assert b in model.bodies
        if charL is not None:
            assert model.charL == charL
        if charT is not None:
            assert model.charT == charT
        if charM is not None:
            assert model.charM == charM

        # check data independence
        if "charL" in quant:
            oldVal = copy(charL)
            charL = 5 * sec
            assert model.charL == oldVal

    @pytest.mark.parametrize(
        "bodies",
        [
            [earth, "Moon"],
            [301, 302],
        ],
    )
    def test_constructorBodyTypeErr(self, bodies):
        with pytest.raises(TypeError):
            DummyModel(bodies)

    @pytest.mark.parametrize(
        "charL, charT, charM",
        [
            (1 * sec, 10 * sec, 10 * kg),
            (1 * km, 1 * km, 10 * kg),
            (1 * km, 1 * sec, 1 * sec),
        ],
    )
    def test_constructorCharQDimErr(self, charL, charT, charM):
        with pytest.raises(pint.DimensionalityError):
            DummyModel(earth, charL, charT, charM)

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
    def test_groupSize(self, model, varGroups, sz):
        assert model.groupSize(varGroups) == sz

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
    def test_extractGroup(self, model, varGroup, shape, out, varIn):
        # standard use case: y has all the variable groups, we want subset out
        y = np.arange(14)
        varOut = model.extractGroup(y, varGroup, varGroupsIn=varIn)
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
    def test_extractGroup_notFull(self, model, y, varIn, varOut, yOut):
        # Test extracting from a vector that doesn't include all the VarGroup
        assert np.array_equal(model.extractGroup(y, varOut, varIn), yOut)

    @pytest.mark.parametrize(
        "y, varIn, varOut",
        [
            [np.arange(2), [VarGroup.STATE], VarGroup.STM],
            [np.arange(2), [VarGroup.STATE], VarGroup.EPOCH_PARTIALS],
            [np.arange(2), [VarGroup.STATE], VarGroup.PARAM_PARTIALS],
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
    def test_extractGroup_missingVarIn(self, model, y, varIn, varOut):
        # varIn doesn't include the desired varOut
        with pytest.raises(RuntimeError):
            model.extractGroup(y, varOut, varIn)

    @pytest.mark.parametrize(
        "y, varIn, varOut",
        [[np.arange(3), [VarGroup.STATE, VarGroup.STM], VarGroup.STM]],
    )
    def test_extractGroup_missingData(self, model, y, varIn, varOut):
        with pytest.raises(ValueError):
            model.extractGroup(y, varOut, varIn)

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

    @pytest.mark.parametrize(
        "wIn, varGroups, wOut",
        [
            # Simple case: single vector
            ([1, 1], [VarGroup.STATE], [[2 * km, 2 * km / (3 * sec)]]),
            # Set of vectors
            (
                [[1, 1], [2, 2], [3, 3]],
                [VarGroup.STATE],
                [
                    [2 * km, 2 * km / (3 * sec)],
                    [4 * km, 4 * km / (3 * sec)],
                    [6 * km, 6 * km / (3 * sec)],
                ],
            ),
            # Transposed vectors are auto-corrected
            (
                [[1, 2, 3], [1, 2, 3]],
                [VarGroup.STATE],
                [
                    [2 * km, 2 * km / (3 * sec)],
                    [4 * km, 4 * km / (3 * sec)],
                    [6 * km, 6 * km / (3 * sec)],
                ],
            ),
            # Multiple variable groups
            (
                [1, 2, 3, 4, 5, 6],
                [VarGroup.STATE, VarGroup.STM],
                [[2 * km, 4 * km / (3 * sec), 3 * UU, 12 * sec, 5 / (3 * sec), 6 * UU]],
            ),
        ],
    )
    def test_toBaseUnits(self, model, wIn, varGroups, wOut):
        w_dim = model.toBaseUnits(wIn, varGroups)
        wOut = np.array(wOut, ndmin=2, dtype=pint.Quantity)

        assert isinstance(w_dim, np.ndarray)
        assert w_dim.shape[1] == model.groupSize(varGroups)
        assert all([isinstance(val, pint.Quantity) for val in w_dim.flat])
        assert all(
            [
                abs(v - vOut).magnitude < 1e-8
                for v, vOut in zip(w_dim.flatten(), wOut.flatten())
            ]
        ), w_dim

    @pytest.mark.parametrize(
        "wOut, varGroups, wIn",
        [
            # Simple case: single vector
            ([1, 1], [VarGroup.STATE], [2 * km, 2 * km / (3 * sec)]),
            # Set of vectors
            (
                [[1, 1], [2, 2], [3, 3]],
                [VarGroup.STATE],
                [
                    [2 * km, 2 * km / (3 * sec)],
                    [4 * km, 4 * km / (3 * sec)],
                    [6 * km, 6 * km / (3 * sec)],
                ],
            ),
            # Transposed vectors are auto-corrected
            (
                [[1, 1], [2, 2], [3, 3]],
                [VarGroup.STATE],
                [
                    [2 * km, 4 * km, 6 * km],
                    [2 * km / (3 * sec), 4 * km / (3 * sec), 6 * km / (3 * sec)],
                ],
            ),
            # Multiple variable groups
            (
                [1, 2, 3, 4, 5, 6],
                [VarGroup.STATE, VarGroup.STM],
                [2 * km, 4 * km / (3 * sec), 3 * UU, 12 * sec, 5 / (3 * sec), 6 * UU],
            ),
        ],
    )
    def test_normalize(self, model, wIn, varGroups, wOut):
        w_nondim = model.normalize(wIn, varGroups)
        wOut = np.array(wOut, ndmin=2)

        assert isinstance(w_nondim, np.ndarray)
        assert w_nondim.shape[1] == model.groupSize(varGroups)
        assert all([isinstance(val, float) for val in w_nondim.flat])
        assert all(
            [
                abs(v - vOut) < 1e-8
                for v, vOut in zip(w_nondim.flatten(), wOut.flatten())
            ]
        ), w_nondim

    def test_eq(self):
        model = DummyModel([earth, sun])
        model2 = DummyModel([earth, sun])

        assert model == model2
        assert not model == "abc"
