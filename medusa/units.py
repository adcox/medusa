"""
Convenient definitions of common units
"""
from . import ureg

# TODO define some type aliases?
# TODO which definition of Quantity to use for types? For construction?

# Meta
Quant = ureg.Quantity

# Length
m = ureg.meter
km = ureg.km
au = ureg.au

# mass
g = ureg.gram
kg = ureg.kg

# time
sec = ureg.sec
minute = ureg.minute
hour = ureg.hour
day = ureg.day
week = ureg.week

# angles
deg = ureg.deg
rad = ureg.rad

# nondimensional coordinates
LU = ureg.LU
TU = ureg.TU
MU = ureg.MU
UU = ureg.Quantity(1, "dimensionless")
