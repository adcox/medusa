"""
Data objects
"""
import xml.etree.ElementTree as ET

# ------------------------------------------------------------------------------
# Constants

GRAV_PARAM = 6.67384e-20
""": Universal gravitational constant (km**3/kg-s**2)"""

G_MEAN_EARTH = 9.80665e-3
""": Mean Earth gravity (km/s**2)"""


class Body:
    """
    Describe a celestial body (star, planet, moon)

    Args:
        name (str): Body name
        gm (float): gravitational parameter (km**2/sec**3)
        sma (Optional, float): orbital semimajor axis (km)
        ecc (Optional, float): orbital eccentricity
        inc (Optional, float): orbital inclination w.r.t. Earth Equatorial J2000 (deg)
        raan (Optional, float): right ascenscion of the ascending node w.r.t.
            Earth Equatorial J2000 (deg)
        spiceId (Optional, int): SPICE ID for this body
        parentId (Optional, int): SPICE ID for the body this body orbits. Set to
            ``None`` if there is no parent body

    Attributes:
        name (str): Body name
        gm (float): gravitational parameter (km**2/sec**3)
        sma (float): orbital semimajor axis (km)
        ecc (float): orbital eccentricity
        inc (float): orbital inclination w.r.t. Earth Equatorial J2000 (deg)
        raan (float): right ascenscion of the ascending node w.r.t.
            Earth Equatorial J2000 (deg)
        spiceId (int): SPICE ID for this body
        parentId (int): SPICE ID for the body this body orbits. Set to
            ``None`` if there is no parent body
    """

    def __init__(
        self, name, gm, sma=0.0, ecc=0.0, inc=0.0, raan=0.0, spiceId=0, parentId=None
    ):
        self.name = name
        self.gm = gm
        self.sma = sma
        self.ecc = ecc
        self.inc = inc
        self.raan = raan
        self.id = spiceId
        self.parentId = parentId

    @staticmethod
    def fromXML(file, name):
        """
        Create a body from an XML file

        Args:
            file (str): path to the XML file
            name (str): Body name

        Returns:
            Body: the corresponding data in a :class:`Body` object. If no body
            can be found that matches ``name``, ``None`` is returned.
        """
        tree = ET.parse(file)
        root = tree.getroot()

        for data in root.iter("body"):
            _name = data.find("name").text
            if _name == name:
                try:
                    pid = int(data.find("parentId").text)
                except:
                    pid = None

                return Body(
                    name,
                    float(data.find("gm").text),
                    sma=float(data.find("circ_r").text),
                    ecc=0.0,
                    inc=float(data.find("inc").text),
                    raan=float(data.find("raan").text),
                    spiceId=int(data.find("id").text),
                    parentId=pid,
                )

        raise RuntimeError(f"Cannot find a body named {name}")

    def __eq__(self, other):
        if not isinstance(other, Body):
            return False

        return (
            self.name == other.name
            and self.gm == other.gm
            and self.sma == other.sma
            and self.ecc == other.ecc
            and self.inc == other.inc
            and self.raan == other.raan
            and self.id == other.id
            and self.parentId == other.parentId
        )
