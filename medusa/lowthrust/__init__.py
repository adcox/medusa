"""
Low thrust module
"""
__all__ = ["control", "dynamics"]

# Enable the following:
#   import medusa.lowthrust as lt
#   obj = lt.control.ConstMassTerm(1.0)
#   model = lt.dynamics.xyz
from . import control, dynamics
