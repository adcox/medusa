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


Full API
=========


Core Classes
-------------

.. automodule:: pika.data
   :members:

.. automodule:: pika.dynamics
   :members:


Circular Restricted Three-Body Problem
--------------------------------------

.. automodule:: pika.dynamics.crtbp
   :members:

