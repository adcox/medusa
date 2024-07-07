"""
Pytest Configuration
"""
import logging
from pathlib import Path

import pytest
from rich.logging import RichHandler

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


# @pytest.fixture(scope="session", autouse=True)
# def logger():
#    """
#    Configure the logger for unit test output
#    """
#    logger = logging.getLogger("pika")
#    logger.handlers.clear()
#    logger.addHandler(RichHandler(show_time=False, enable_link_path=False))
#    logger.setLevel(logging.DEBUG)
#    logger.propagate = False
#    return logger.name
