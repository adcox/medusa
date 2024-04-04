"""
Pytest Configuration
"""
from pathlib import Path

BODY_XML = Path(__file__).parent / "../resources/body-data.xml"


def loadBody(name):
    """
    Convenience function to return body from the default XML file

    Args:
        name (str): body name

    Returns:
        Body: the body object
    """
    from pika.data import Body

    return Body.fromXML(BODY_XML, name)
