"""
Test basic dynamics
"""
from collections.abc import Sequence
from copy import copy, deepcopy

import numpy as np
import pint
import pytest
from conftest import loadBody

from medusa import util
from medusa.dynamics import DynamicsModel, State, VarGroup
from medusa.units import LU, MU, TU, UU, kg, km, sec

earth, moon, sun = loadBody("Earth"), loadBody("Moon"), loadBody("Sun")


# ------------------------------------------------------------------------------
# Simple implementations of abstract objects for testing
# ------------------------------------------------------------------------------
class DummyModel(DynamicsModel):
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

    def makeState(self, data, time, center, frame):
        return DummyState(self, data, time, center, frame)


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

    def groupSize(self, varGroups):
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
class TestDynamicsModel:
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

        preReg = DynamicsModel._registry.copy()
        model = DummyModel(bodies, **quant)
        assert len(DynamicsModel._registry) - len(preReg) == 1

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
        # Data that hasn't been passed in is equal to NaN
        assert all(np.isnan(state._data[ix]) for ix in range(len(data), 14))

        # ensure copy by value
        d0 = data[0]
        data[0] *= 2
        assert state._data[0] == d0

    @pytest.mark.parametrize(
        "modelOverride, data, err",
        [
            (None, [2.0], ValueError),
            (None, np.arange(15), ValueError),
            ("abc", [1.0, 2.0], TypeError),
        ],
    )
    def test_constructor_errs(self, model, modelOverride, data, err):
        if modelOverride is not None:
            model = modelOverride

        with pytest.raises(err):
            DummyState(model, data)

    def test_deepcopy(self, model):
        state = DummyState(model, [1, 2])
        state2 = deepcopy(state)

        # Model should not be copied due to Mixin that prevents it
        assert id(state.model) == id(state2.model)

    def test_arrayLike_get(self, model):
        data = [1, 2, 3, 4, 5, 6]
        state = DummyState(model, data)

        # Individual indices
        for ix in range(len(data)):
            assert state[ix] == data[ix]

        # Slices
        np.testing.assert_equal(state[:3], data[:3])

    def test_arrayLike_set(self, model):
        data = [1, 2]
        state = DummyState(model, data)

        # Individual indices
        state[0] = 1
        assert state._data[0] == 1

        # Slices
        state[:2] = [4, 5]
        np.testing.assert_equal(state._data[:2], [4, 5])

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
    def testgroupSize(self, model, varGroups, sz):
        state = DummyState(model, [1.0, 2.0])
        assert state.groupSize(varGroups) == sz

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
    @pytest.mark.parametrize("vals", [None, np.arange(14).tolist()])
    def test_extractGroup(self, model, varGroup, vals, shape, out):
        # Standard use case: get subset of state data
        state = DummyState(model, np.arange(14))
        varOut = state._extractGroup(varGroup, vals=vals)
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
        "groups, out, reshape",
        [
            # Simple cases: extract one vargroup
            (VarGroup.STATE, [0, 1], False),
            (VarGroup.STM, [2, 3, 4, 5], False),
            (VarGroup.EPOCH_PARTIALS, [6, 7], False),
            (VarGroup.PARAM_PARTIALS, [8, 9, 10, 11, 12, 13], False),
            (VarGroup.STATE, [0, 1], True),
            (VarGroup.STM, [[2, 3], [4, 5]], True),
            (VarGroup.PARAM_PARTIALS, [[8, 9, 10], [11, 12, 13]], True),
            # Extract multiple, not necessarily in order
            ([VarGroup.STATE, VarGroup.STM], [0, 1, 2, 3, 4, 5], False),
            ([VarGroup.STATE, VarGroup.EPOCH_PARTIALS], [0, 1, 6, 7], False),
            ([VarGroup.STM, VarGroup.EPOCH_PARTIALS], [2, 3, 4, 5, 6, 7], False),
            ([VarGroup.STM, VarGroup.STATE], [2, 3, 4, 5, 0, 1], False),
            ([VarGroup.STATE, VarGroup.STM], ([0, 1], [[2, 3], [4, 5]]), True),
        ],
    )
    def test_get(self, model, groups, out, reshape):
        state = DummyState(model, np.arange(14))
        vals = state.get(groups, reshape=reshape)
        np.testing.assert_equal(vals, out)

    @pytest.mark.parametrize(
        "groups, data, indices",
        [
            # Set a single variable group
            (VarGroup.STATE, [98, 99], [0, 1]),
            (VarGroup.STM, [1, 2, 3, 4], [2, 3, 4, 5]),
            (VarGroup.EPOCH_PARTIALS, [0, 1], [6, 7]),
            (VarGroup.PARAM_PARTIALS, 100 + np.arange(6), 8 + np.arange(6)),
            # Set multiple
            ([VarGroup.STATE, VarGroup.STM], 100 + np.arange(6), np.arange(6)),
        ],
    )
    def test_set(self, model, groups, data, indices):
        state = DummyState(model, np.arange(14))
        state.set(groups, data)

        for count, ix in enumerate(indices):
            assert state[ix] == data[count]
        for ix in range(14):
            if not ix in indices:
                assert state[ix] == ix

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

        # The default ICs should be in the data now
        groupVals = state.get(sorted(util.toList(groups)), reshape=False)
        np.testing.assert_equal(groupVals, appended)

        # Other values should be unchanged
        for group in State.ALL_VARS:
            if not group in util.toList(groups):
                np.testing.assert_equal(
                    state.get(group, reshape=False),
                    state._extractGroup(group, vals=data, reshape=False),
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
        assert w_dim.shape[1] == state.groupSize(varGroups)
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
        assert w_nondim.shape[1] == state.groupSize(varGroups)
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

    @pytest.mark.parametrize(
        "obj, out",
        [
            (1, [2, 3] + [np.nan] * 12),
            (np.arange(14), [1, 3] + [np.nan] * 12),
            ([0, 1], [1, 3] + [np.nan] * 12),
            (np.asarray([0, 1]), [1, 3] + [np.nan] * 12),
        ],
    )
    def test_add_floats(self, model, obj, out):
        state = DummyState(model, [1, 2])
        state2 = state + obj
        np.testing.assert_equal(state2._data, out)

    @pytest.mark.parametrize(
        "vals, kwargs, out",
        [
            ([1, 1], {}, [2, 3] + [np.nan] * 12),
            ([1, 1], {"center": "earth"}, RuntimeError),
            ([1, 1], {"frame": "inertial"}, RuntimeError),
        ],
    )
    def test_add_states(self, model, vals, kwargs, out):
        state = DummyState(model, [1, 2])
        state2 = DummyState(model, vals, **kwargs)
        if isinstance(out, Sequence):
            state3 = state + state2
            np.testing.assert_equal(state3._data, out)
        else:
            with pytest.raises(out):
                state + state2

    @pytest.mark.parametrize(
        "vals, out", [(1, True), (2, True), (3, False), (np.nan, False)]
    )
    def test_contains(self, model, vals, out):
        state = DummyState(model, [1, 2])
        assert (vals in state) == out

    @pytest.mark.parametrize("vals, out", [([1, 2], 14), (np.arange(4), 14)])
    def test_len(self, model, vals, out):
        state = DummyState(model, vals)
        assert len(state) == out

    @pytest.mark.parametrize(
        "vals, out", [(3, [3, 6] + [np.nan] * 12), ([2, 1], [2, 2] + [np.nan] * 12)]
    )
    def test_mul(self, model, vals, out):
        state = DummyState(model, [1, 2])
        state2 = state * vals
        np.testing.assert_equal(state2._data, out)

    @pytest.mark.parametrize(
        "vals, out",
        [
            (np.eye(2), [1, 2] + [np.nan] * 12),
            (2 * np.eye(2), [2, 4] + [np.nan] * 12),
            (np.full((2, 2), 1), [3, 3] + [np.nan] * 12),
            (np.eye(3), [np.nan] * 14),
        ],
    )
    def test_matmul(self, model, vals, out):
        state = DummyState(model, [1, 2])
        state2 = state @ vals
        np.testing.assert_equal(state2._data, out)

    def test_neg(self, model):
        state = DummyState(model, np.arange(14))
        state2 = -state
        np.testing.assert_equal(state2._data, -(state._data))

    @pytest.mark.parametrize(
        "obj, out",
        [
            (1, [0, 1] + [np.nan] * 12),
            (np.arange(14), [1, 1] + [np.nan] * 12),
            ([0, 1], [1, 1] + [np.nan] * 12),
            (np.asarray([0, 1]), [1, 1] + [np.nan] * 12),
        ],
    )
    def test_sub_floats(self, model, obj, out):
        state = DummyState(model, [1, 2])
        state2 = state - obj
        np.testing.assert_equal(state2._data, out)

    @pytest.mark.parametrize(
        "vals, kwargs, out",
        [
            ([1, 1], {}, [0, 1] + [np.nan] * 12),
            ([1, 1], {"center": "earth"}, RuntimeError),
            ([1, 1], {"frame": "inertial"}, RuntimeError),
        ],
    )
    def test_sub_states(self, model, vals, kwargs, out):
        state = DummyState(model, [1, 2])
        state2 = DummyState(model, vals, **kwargs)
        if isinstance(out, Sequence):
            state3 = state - state2
            np.testing.assert_equal(state3._data, out)
        else:
            with pytest.raises(out):
                state - state2
