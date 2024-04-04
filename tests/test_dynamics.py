"""
Test dynamics
"""

from pika.dynamics import EOMVars


def test_eomVars():
    assert EOMVars.STATE in EOMVars.ALL
    assert EOMVars.STM in EOMVars.ALL
    assert EOMVars.EPOCH_DEPS in EOMVars.ALL
    assert EOMVars.PARAM_DEPS in EOMVars.ALL
