===========
medusa API
===========

.. module:: medusa

.. toctree::
   :maxdepth: 2
   :titlesonly:

.. contents:: API Contents
   :depth: 3
   :local:

Overview
=========

The library includes functionality to model the dynamics of various models, 
perform numerical integration, and run numerical corrections processes.

Dynamics
--------
The core of the library is a set of classes that model the dynamics of the
motion of bodies in space. 

.. autosummary::
   :nosignatures:

   ~data.Body
   ~dynamics.AbstractDynamicsModel
   ~dynamics.EOMVars

Some of these objects are purly abstract classes and cannot be instantiated.
Individual models extend the base classes to provide concrete implementations.

.. autosummary::
   :nosignatures:

   crtbp.DynamicsModel
   lowthrust.dynamics.LowThrustCrtbpDynamics

The low-thrust model defines additional objects that specify the acceleration 
that is added to the dynamics.

.. autosummary::
   :nosignatures:

   lowthrust.control.ControlTerm
   lowthrust.control.ConstThrustTerm
   lowthrust.control.ConstMassTerm
   lowthrust.control.ConstOrientTerm
   lowthrust.control.ControlLaw
   lowthrust.control.SeparableControlLaw
   lowthrust.control.ForceMassOrientLaw


Propagation
-----------

The propagation code includes objects to perform the propagation.

.. autosummary::
   :nosignatures:

   ~propagate.Propagator


Event objects can also be added to the propagation to control stop conditions
and/or mark relevant times in the propagation.

.. autosummary::
   :nosignatures:

   ~propagate.AbstractEvent
   ~propagate.ApseEvent
   ~propagate.BodyDistanceEvent
   ~propagate.DistanceEvent
   ~propagate.VariableValueEvent

Corrections
-----------

Numerical corrections (e.g., single shooting, multiple shooting) problems
contain variables, sometimes packaged into more complex data
structures such as propagated segments and/or control points.

.. autosummary::
   :nosignatures:

   ~corrections.CorrectionsProblem
   ~corrections.ShootingProblem
   ~corrections.Variable
   ~corrections.ControlPoint
   ~corrections.Segment

A variety of constraints can be added to the corrections problem.

.. autosummary::
   :nosignatures:

   ~corrections.AbstractConstraint
   ~corrections.constraints.ContinuityConstraint
   ~corrections.constraints.VariableValueConstraint

Corrections problems are solved by a differential corrector.

.. autosummary::
   :nosignatures:

   ~corrections.DifferentialCorrector
   ~corrections.MinimumNormUpdate
   ~corrections.LeastSquaresUpdate
   ~corrections.L2NormConvergence

Full API
=========


Core Classes
-------------

.. automodule:: medusa.data
   :members:
   :show-inheritance:

.. automodule:: medusa.dynamics
   :members:
   :show-inheritance:


.. automodule:: medusa.propagate
   :members:
   :show-inheritance:

.. automodule:: medusa.corrections
   :members:
   :show-inheritance:

.. automodule:: medusa.corrections.constraints
   :members:
   :show-inheritance:



Circular Restricted Three-Body Problem
--------------------------------------

.. automodule:: medusa.crtbp
   :members:
   :show-inheritance:


Low-Thrust Modeling
--------------------

.. automodule:: medusa.lowthrust.dynamics
   :members:
   :show-inheritance:

.. automodule:: medusa.lowthrust.control
   :members:
   :show-inheritance:

Utilities
----------------

.. automodule:: medusa.numerics
   :members:
   :show-inheritance:

.. automodule:: medusa.plots
   :members:
   :show-inheritance:

.. automodule:: medusa.util
   :members:
   :show-inheritance:
