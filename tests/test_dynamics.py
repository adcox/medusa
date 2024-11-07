"""
Test basic dynamics
"""
from copy import copy

import numpy as np
import pint
import pytest
from conftest import loadBody

from medusa import util
from medusa.dynamics import AbstractDynamicsModel, State, VarGroup
from medusa.units import LU, MU, TU, UU, kg, km, sec

earth, moon, sun = loadBody("Earth"), loadBody("Moon"), loadBody("Sun")


# ------------------------------------------------------------------------------
# Simple implementations of abstract objects for testing
# ------------------------------------------------------------------------------
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


class DummyState(State):
    def __init__(self, model, data, time=0.0, center="center", frame="frame"):
        super().__init__(model, data, time, center, frame)

    def units(self, varGroup):
        if varGroup == VarGroup.STATE:
            # let's say the state is a position and a velocity
            return [LU, LU / TU]
        elif varGroup == VarGroup.STM:
            # STM units are [1, time, 1/time, 1]
            return [UU, TU, UU / TU, UU]
        elif varGroup == VarGroup.EPOCH_PARTIALS:
            return [UU / TU, UU / TU]  # completely fictional
        elif varGroup == VarGroup.PARAM_PARTIALS:
            return [MU, MU / TU, MU * LU / TU**2] * 2  # completely fictional

    def _groupSize(self, varGroups):
        # A model with 2 state variables, epoch dependencies, and three parameters
        varGroups = np.array(varGroups, ndmin=1)
        return (
            2 * (VarGroup.STATE in varGroups)
            + 4 * (VarGroup.STM in varGroups)
            + 2 * (VarGroup.EPOCH_PARTIALS in varGroups)
            + 6 * (VarGroup.PARAM_PARTIALS in varGroups)
        )


# ------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------
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

    def test_repr(self, model):
        assert isinstance(repr(model), str)

    def test_eq(self):
        model = DummyModel([earth, sun])
        model2 = DummyModel([earth, sun])

        assert model == model2
        assert not model == "abc"


class TestState:
    @pytest.fixture(scope="class")
    def model(self):
        return DummyModel([sun, earth, moon])

    @pytest.mark.parametrize(
        "data",
        [
            [1.0, 2.0],
            np.array([1.0, 2.0], ndmin=2),
            np.arange(1, 7, dtype=float),
            np.arange(1, 15, dtype=float),
        ],
    )
    @pytest.mark.parametrize("time, center, frame", [(3.0, "Earth", "Rotating")])
    def test_constructor(self, model, data, time, center, frame):
        state = DummyState(model, data, time, center, frame)

        assert state.time == time
        assert state.center == center
        assert state.frame == frame

        data = np.asarray(data).flatten()  # for comparison sake, make 1D
        assert isinstance(state._data, np.ndarray)
        # All variables have a slot
        assert state._data.shape == (14,)
        # The data passed in has been copied
        np.testing.assert_equal(state._data[: len(data)], data)
        # Data that hasn't been passed in is equal to zero
        assert all(state._data[ix] == 0.0 for ix in range(len(data), 14))

        # ensure copy by value
        d0 = data[0]
        data[0] *= 2
        assert state._data[0] == d0

    @pytest.mark.parametrize(
        "modelOverride, data, err",
        [
            (None, [2.0], ValueError),
            ("abc", [1.0, 2.0], TypeError),
        ],
    )
    def test_constructor_errs(self, model, modelOverride, data, err):
        if modelOverride is not None:
            model = modelOverride

        with pytest.raises(err):
            DummyState(model, data)

    def test_arrayLike(self, model):
        data = [1, 2, 3, 4, 5, 6]
        state = DummyState(model, data)

        for ix in range(len(data)):
            assert state[ix] == data[ix]

    # Sanity check for my testing implementation...
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
        state = DummyState(model, [1.0, 2.0])
        assert state._groupSize(varGroups) == sz

    @pytest.mark.parametrize(
        "varGroup, shape, out",
        [
            (VarGroup.STATE, (2,), [0, 1]),
            (VarGroup.STATE, (2,), [0, 1]),
            (VarGroup.STM, (2, 2), [[2, 3], [4, 5]]),
            (VarGroup.EPOCH_PARTIALS, (2,), [6, 7]),
            (VarGroup.PARAM_PARTIALS, (2, 3), [[8, 9, 10], [11, 12, 13]]),
        ],
    )
    def test_extractGroup(self, model, varGroup, shape, out):
        # Standard use case: get subset of state data
        state = DummyState(model, np.arange(14))
        varOut = state._extractGroup(varGroup)
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
        state = DummyState(model, [1, 2])
        # Test extracting from a vector that doesn't include all the VarGroup
        assert np.array_equal(state._extractGroup(varOut, y, varIn), yOut)

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
        state = DummyState(model, [1, 2])
        # varIn doesn't include the desired varOut
        with pytest.raises(RuntimeError):
            state._extractGroup(varOut, y, varIn)

    @pytest.mark.parametrize(
        "y, varIn, varOut",
        [[np.arange(3), [VarGroup.STATE, VarGroup.STM], VarGroup.STM]],
    )
    def test_extractGroup_missingData(self, model, y, varIn, varOut):
        state = DummyState(model, [1, 2])
        with pytest.raises(ValueError):
            state._extractGroup(varOut, y, varIn)

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
        state = DummyState(model, [1, 2])
        assert state._defaultICs(VarGroup).tolist() == out

    @pytest.mark.parametrize(
        "groups, appended",
        [
            [VarGroup.STM, [1, 0, 0, 1]],
            [VarGroup.EPOCH_PARTIALS, [0] * 2],
            [VarGroup.PARAM_PARTIALS, [0] * 6],
            [VarGroup.STATE, [0] * 2],
            [[VarGroup.STM, VarGroup.EPOCH_PARTIALS], [1, 0, 0, 1, 0, 0]],
            [[VarGroup.EPOCH_PARTIALS, VarGroup.STM], [1, 0, 0, 1, 0, 0]],
        ],
    )
    def test_fillDefaultICs(self, model, groups, appended):
        data = 1 + np.arange(14)  # fill with non-default values
        state = DummyState(model, data)
        state.fillDefaultICs(groups)

        groupVals = []
        for group in sorted(util.toList(groups)):
            groupVals.extend(state._extractGroup(group).flatten())

        # The default ICs should be in the data now
        np.testing.assert_equal(groupVals, appended)

        # Other values should be unchanged
        for group in State.ALL_VARS:
            if not group in util.toList(groups):
                np.testing.assert_equal(
                    state._extractGroup(group), state._extractGroup(group, vals=data)
                )

    @pytest.mark.parametrize(
        "grp, out",
        [
            (VarGroup.STATE, ["State 0", "State 1"]),
            (VarGroup.STM, ["STM(0,0)", "STM(0,1)", "STM(1,0)", "STM(1,1)"]),
            (VarGroup.EPOCH_PARTIALS, ["Epoch Dep 0", "Epoch Dep 1"]),
            (
                VarGroup.PARAM_PARTIALS,
                [f"Param Dep({r},{c})" for r in range(2) for c in range(3)],
            ),
        ],
    )
    def test_coords(self, model, grp, out):
        state = DummyState(model, [1, 2])
        assert state.coords(grp) == out

    @pytest.mark.parametrize("grp", [7, -1])
    def test_coords_invalid(self, model, grp):
        state = DummyState(model, [1, 2])
        with pytest.raises(ValueError):
            state.coords(grp)

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
        state = DummyState(model, [1, 2])
        w_dim = state.toBaseUnits(wIn, varGroups)
        wOut = np.array(wOut, ndmin=2, dtype=pint.Quantity)

        assert isinstance(w_dim, np.ndarray)
        assert w_dim.shape[1] == state._groupSize(varGroups)
        assert all([isinstance(val, pint.Quantity) for val in w_dim.flat])
        assert all(
            [
                abs(v - vOut).magnitude < 1e-8
                for v, vOut in zip(w_dim.flatten(), wOut.flatten())
            ]
        ), w_dim

    @pytest.mark.parametrize(
        "wIn, varGroups, err",
        [
            ([1 * km, 1 * km], [VarGroup.STATE], TypeError),
            ([1, 2, 3], [VarGroup.STATE], ValueError),
        ],
    )
    def test_toBaseUnits_errs(self, model, wIn, varGroups, err):
        state = DummyState(model, [1, 2])
        with pytest.raises(err):
            state.toBaseUnits(wIn, varGroups)

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
        state = DummyState(model, [1, 2])
        w_nondim = state.normalize(wIn, varGroups)
        wOut = np.array(wOut, ndmin=2)

        assert isinstance(w_nondim, np.ndarray)
        assert w_nondim.shape[1] == state._groupSize(varGroups)
        assert all([isinstance(val, float) for val in w_nondim.flat])
        assert all(
            [
                abs(v - vOut) < 1e-8
                for v, vOut in zip(w_nondim.flatten(), wOut.flatten())
            ]
        ), w_nondim

    @pytest.mark.parametrize(
        "wIn, varGroups, err",
        [
            ([1, 1 * km], [VarGroup.STATE], TypeError),
            ([1 * km, 2 * km, 3 * km], [VarGroup.STATE], ValueError),
        ],
    )
    def test_normalize_errs(self, model, wIn, varGroups, err):
        state = DummyState(model, [1, 2])
        with pytest.raises(err):
            state.normalize(wIn, varGroups)
