"""
Data
====

This module defines several constants,

.. autosummary::
   :nosignatures:

   GRAV_PARAM
   G_MEAN_EARTH

A :class:`Body` class is also defined to store data about a celestial object
such as a planet, moon, or star.


Reference
---------

.. autodata:: GRAV_PARAM
.. autodata:: G_MEAN_EARTH

.. autoclass:: Body
   :members:

"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Union

import pint

from .units import Quant, deg, kg, km, sec
from .util import float_eq

# ------------------------------------------------------------------------------
# Constants

GRAV_PARAM = 6.67384e-20 * km**3 / (kg * sec**2)
""": Universal gravitational constant"""

G_MEAN_EARTH = 9.80665e-3 * km / (sec**2)
""": Mean Earth gravity"""


class Body:
    """
    Describe a celestial body (star, planet, moon)

    Args:
        name: Body name
        gm: gravitational parameter
        sma: orbital semimajor axis
        ecc: orbital eccentricity
        inc: orbital inclination w.r.t. Earth Equatorial J2000
        raan: right ascenscion of the ascending node w.r.t.
            Earth Equatorial J2000
        spiceId: SPICE ID for this body
        parentId: SPICE ID for the body this body orbits. Set to
            ``None`` if there is no parent body
    """

    def __init__(
        self,
        name: str,
        gm: pint.Quantity,
        sma: pint.Quantity = 0.0 * km,
        ecc: pint.Quantity = Quant(0.0),
        inc: pint.Quantity = 0.0 * deg,
        raan: pint.Quantity = 0.0 * deg,
        spiceId: int = 0,
        parentId: Union[int, None] = None,
    ) -> None:
        #: Body name
        self.name: str = name

        #: Gravitational parameter (km**2/sec**3)
        self.gm: pint.Quantity = gm

        #: orbital semimajor axis (km)
        self.sma: pint.Quantity = sma

        #: orbital eccentricity
        self.ecc: pint.Quantity = ecc

        #: orbital inclination w.r.t. Earth equatorial J2000 (deg)
        self.inc: pint.Quantity = inc

        #: right ascension of the ascending node w.r.t. Earth equatorial J2000 (deg)
        self.raan: pint.Quantity = raan

        #: SPICE ID for this body
        self.id: int = spiceId

        #: SPICE ID for body this one orbits; set to ``None`` if there is no parent
        self.parentId = parentId

    @staticmethod
    def fromXML(file: str, name: str) -> Body:
        """
        Create a body from an XML file

        Args:
            file: path to the XML file
            name: Body name

        Returns:
            the corresponding data in a :class:`Body` object. If no body
            can be found that matches ``name``, ``None`` is returned.
        """
        tree = ET.parse(file)
        root = tree.getroot()

        def _expect(data: ET.Element, key: str) -> str:
            # Get value from data element, raise exception if it isn't there
            obj = data.find(key)
            if obj is None:
                raise RuntimeError(f"Could not read '{key}' for {name}")
            else:
                return str(obj.text)

        for data in root.iter("body"):
            obj = data.find("name")
            if obj is None:
                continue

            _name = obj.text
            if _name == name:
                try:
                    pid = int(data.find("parentId").text)  # type: ignore
                except:
                    pid = None

                return Body(
                    name,
                    Quant(float(_expect(data, "gm")), "km**3 / sec**2"),
                    sma=Quant(float(_expect(data, "circ_r")), "km"),
                    ecc=Quant(0.0),  # TODO why is this not read from data??
                    inc=Quant(float(_expect(data, "inc")), "deg"),
                    raan=Quant(float(_expect(data, "raan")), "deg"),
                    spiceId=int(_expect(data, "id")),
                    parentId=pid,
                )

        raise RuntimeError(f"Cannot find a body named {name}")

    def __eq__(self, other):
        if not isinstance(other, Body):
            return False

        return (
            self.name == other.name
            and float_eq(self.gm, other.gm)
            and float_eq(self.sma, other.sma)
            and float_eq(self.ecc, other.ecc)
            and float_eq(self.inc, other.inc)
            and float_eq(self.raan, other.raan)
            and self.id == other.id
            and self.parentId == other.parentId
        )

    def __repr__(self):
        vals = ", ".join(
            [
                "{!s}={!r}".format(lbl, getattr(self, lbl))
                for lbl in ("gm", "sma", "ecc", "inc", "raan", "id", "parentId")
            ]
        )
        return "<Body {!s}: {!s}>".format(self.name, vals)
