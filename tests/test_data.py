"""
Test Body Data
"""
import pytest
from conftest import BODY_XML

from medusa.data import Body
from medusa.units import Quant, deg, km, rad, sec

# ------------------------------------------------------------------------------
# Tests for Body


def test_repr():
    body = Body("Vega", gm=3e55 * km**2 / sec**3)
    assert isinstance(repr(body), str)


def test_constructor_min():
    name = "Midgard"
    gm = 5.6e5 * km**2 / sec**3

    body = Body(name, gm)
    assert body.name == name
    assert body.gm == gm
    for attr in ["sma", "ecc", "inc", "raan"]:
        assert getattr(body, attr) == 0.0
    assert body.id == 0
    assert body.parentId is None


def test_constructor_opts():
    opts = {
        "sma": 1.0e5 * km,
        "ecc": 0.3,
        "inc": 0.1 * rad,
        "raan": 0.6 * rad,
        "spiceId": 3,
        "parentId": 10,
    }
    body = Body("Test", 3.2e5 * km**2 / sec**3, **opts)

    for opt, val in opts.items():
        attr = "id" if opt == "spiceId" else opt
        assert getattr(body, attr) == val


@pytest.mark.parametrize("name", ["Sun", "Earth", "Moon", "Earth Barycenter", "Triton"])
def test_readXML(name):
    body = Body.fromXML(BODY_XML, name)
    assert isinstance(body.name, str)
    assert body.name == name

    for attr in ("gm", "sma", "ecc", "inc", "raan"):
        assert isinstance(getattr(body, attr), Quant)

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
        ("Earth", "Earth", True),
        ("Earth", "Moon", False),
        ("Sun", "Sun", True),
        ("Solar System Barycenter", "Solar System Barycenter", True),
        ("Sun", "Solar System Barycenter", False),
    ],
)
def test_equal(b1, b2, tf):
    b1 = Body.fromXML(BODY_XML, b1)
    b2 = Body.fromXML(BODY_XML, b2)
    assert (b1 == b2) == tf


def test_equal_other():
    body = Body.fromXML(BODY_XML, "Europa")
    assert not body == "Europa"
