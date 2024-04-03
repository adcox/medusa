"""
Data objects
"""
import xml.etree.ElementTree as ET


class Body:
    def __init__(
        self, name, gm, sma=0.0, ecc=0.0, inc=0.0, raan=0.0, spiceId=0, parentId=0
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

        return None
