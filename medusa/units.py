"""
Units
======

Medusa relies on the :mod:`pint` library for unit management. A **unit registry**
is defined in the root medusa package.

.. autosummary:: medusa.ureg

This registry holds the entire collection of maneuver definitions,

.. code-block:: python

   from medusa import ureg

   distance = 1 * ureg.km
   time = 2 * ureg.sec

Pint supports many different units with multiple aliases for each unit. For instance,
``ureg.km`` and ``ureg.kilometer`` are both valid definitions. See the 
`Pint Quantity`_ docs for more information.

For convenience, this module defines aliases, listed below, to the most commonly
used units in the registry. Others can be accessed via :data:`~medusa.ureg`.

**Length**

.. autosummary::
   mm
   m
   km
   au

**Mass**

.. autosummary::
   g
   kg

**Time**

.. autosummary::
   sec
   minute
   hour
   day
   week

**Angle**

.. autosummary::
   deg
   rad

.. _Pint Quantity: https://pint.readthedocs.io/en/stable/user/defining-quantities.html

Normalized Units
-----------------

When operating on the scale of the solar system,
quantities can have very different magnitudes. For example, a spacecraft may be
1.5e6 km away from the Sun and while traveling at 0.5 km/sec. A calculation that
includes both of these calculations can suffer from round-off errors because of 
the different orders of magnitude.

To mitigate numerical issues, almost all of the calculations within ``medusa``
operate on data that have been **normalized** by three fundamental "characteristic
quantities,"

.. autosummary::
   LU
   TU
   MU

Each dynamical model defines its own values for ``LU``, ``TU``, and ``MU`` so that
the normalized values in these units can easily be transcribed to the usual physical
units (km, sec, kg, etc.). The :func:`AbstractDynamicsModel.toBaseUnits` and
:func:`AbstractDynamicsModel.normalize` functions make converting between normalized
and standard units easy.

Working with Arrays
-------------------

Units and Quantities can be stored in numpy arrays just like other numeric types.
However, some array construction techniques don't quite know how to extrapolate
Quantity values. In these cases, the fix is usually to define the array's ``dtype``,

.. code-block:: python

   import numpy as np
   from medusa import ureg
   from medusa.units import km, sec

   np.array([1*km, 2*sec], ndmin=2)                         # Fails
   np.array([1*km, 2*sec], ndmin=2, dtype=object)           # Ok
   np.array([1*km, 2*sec], ndmin=2, dtype=ureg.Quantity)    # Ok

If an array contains values of only one unit, it can be defined as a quantity 
instead of an array of Quantities,

.. code-block:: python

   ureg.Quantity([1, 2], "km")

For more information, see the `Pint NumPy`_ docs.

.. _Pint NumPy: https://pint.readthedocs.io/en/stable/user/numpy.html

Reference
---------

.. autodata:: medusa.ureg
.. autodata:: mm
.. autodata:: m
.. autodata:: km
.. autodata:: au
.. autodata:: g
.. autodata:: kg
.. autodata:: sec
.. autodata:: minute
.. autodata:: hour
.. autodata:: day
.. autodata:: week
.. autodata:: deg
.. autodata:: rad
.. autodata:: LU
.. autodata:: TU
.. autodata:: MU
.. autodata:: UU
"""

# ------------------------------------------------------------------------------
# Meta
from . import ureg

Quant = ureg.Quantity

# ------------------------------------------------------------------------------
# Length
mm = ureg.mm  #: one milimeter
m = ureg.meter  #: one meter
km = ureg.km  #: one kilometer
au = ureg.au  #: one astronomical unit

# ------------------------------------------------------------------------------
# mass
g = ureg.gram  #: one gram
kg = ureg.kg  #: one kilogram

# ------------------------------------------------------------------------------
# time
sec = ureg.sec  #: one second
minute = ureg.minute  #: one minute
hour = ureg.hour  #: one hour
day = ureg.day  #: one day
week = ureg.week  #: one week

# ------------------------------------------------------------------------------
# angles
deg = ureg.deg  #: one degree
rad = ureg.rad  #: one radian

# ------------------------------------------------------------------------------
# nondimensional coordinates
LU = ureg.LU  #: length unit for dynamical models

TU = ureg.TU  #: time unit for dynamical models

MU = ureg.MU  #: mass unit for dynamical models

UU = ureg.Unit("dimensionless")  #: shorthand for a dimensionless quantity
