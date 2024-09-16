"""
Test Body Data
"""
import pytest
from conftest import BODY_XML

from medusa.data import Body

# ------------------------------------------------------------------------------
# Tests for Body


def test_constructor_min():
    name = "Midgard"
    gm = 5.6e5

    body = Body(name, gm)
    assert body.name == name
    assert body.gm == gm
    for attr in ["sma", "ecc", "inc", "raan"]:
        assert getattr(body, attr) == 0.0
    assert body.id == 0
    assert body.parentId is None


def test_constructor_opts():
    opts = {
        "sma": 1.0e5,
        "ecc": 0.3,
        "inc": 0.1,
        "raan": 0.6,
        "spiceId": 3,
        "parentId": 10,
    }
    body = Body("Test", 3.2e5, **opts)

    for opt, val in opts.items():
        attr = "id" if opt == "spiceId" else opt
        assert getattr(body, attr) == val


@pytest.mark.parametrize("name", ["Sun", "Earth", "Moon", "Earth Barycenter", "Triton"])
def test_readXML(name):
    body = Body.fromXML(BODY_XML, name)
    assert isinstance(body.name, str)
    assert body.name == name

    for attr in ("gm", "sma", "ecc", "inc", "raan"):
        assert isinstance(getattr(body, attr), float)

    assert isinstance(body.id, int)
    assert isinstance(body.parentId, int) or body.parentId is None


def test_readXML_notFound():
    with pytest.raises(RuntimeError):
        Body.fromXML(BODY_XML, "Kolob")


earth = Body.fromXML(BODY_XML, "Earth")
moon = Body.fromXML(BODY_XML, "Moon")
sun = Body.fromXML(BODY_XML, "Sun")
ssb = Body.fromXML(BODY_XML, "Solar System Barycenter")


@pytest.mark.parametrize(
    "b1, b2, tf",
    [
        (earth, earth, True),
        (earth, moon, False),
        (sun, sun, True),
        (ssb, ssb, True),
        (sun, ssb, False),
        (earth, "Earth", False),
    ],
)
def test_equal(b1, b2, tf):
    assert (b1 == b2) == tf