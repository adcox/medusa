===========
Pika API
===========

.. module:: pika

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
   ~dynamics.ModelConfig
   ~dynamics.EOMVars

Some of these objects are purly abstract classes and cannot be instantiated.
Individual models extend the base classes to provide concrete implementations.

.. autosummary::
   :nosignatures:

   dynamics.crtbp.DynamicsModel
   dynamics.crtbp.ModelConfig


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

Numerical corrections (e.g., single shooting, multiple shooting) leverage common
base classes with derived classes for specific solution algorithms.

.. autosummary::
   :nosignatures:

   ~corrections.CorrectionsProblem
   ~corrections.Variable

A variety of constraints can be added to the corrections problem.

.. autosummary::
   :nosignatures:

   ~corrections.AbstractConstraint
   ~corrections.constraints.VariableValueConstraint


Full API
=========


Core Classes
-------------

.. automodule:: pika.data
   :members:

.. automodule:: pika.dynamics
   :members:

.. automodule:: pika.propagate
   :members:

.. automodule:: pika.corrections
   :members:

.. automodule:: pika.corrections.constraints
   :members:


Circular Restricted Three-Body Problem
--------------------------------------

.. automodule:: pika.dynamics.crtbp
   :members:

