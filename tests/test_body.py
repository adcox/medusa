"""
Test Body Data
"""
from pathlib import Path

import pytest

from pika.data import Body

XML = Path(__file__).parent / "../resources/body-data.xml"


def test_constructor_min():
    name = "Midgard"
    gm = 5.6e5

    body = Body(name, gm)
    assert body.name == name
    assert body.gm == gm
    for attr in ["sma", "ecc", "inc", "raan", "id", "parentId"]:
        assert getattr(body, attr) == 0.0


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
    body = Body.fromXML(XML, name)
    assert body.name == name
