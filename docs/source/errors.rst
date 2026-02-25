Errors
======

GenPlanner defines a small hierarchy of custom exceptions.
All errors inherit from :class:`GenPlannerBaseError`.

The base class supports an optional ``run_name`` attribute,
which is included in the error message when present.

Example:

.. code-block:: python

   try:
       ...
   except Exception as e:
       print(e)

If a run name is set, the message format becomes:

::

   [gp_240226_14_32] error message here


------------------------------------------------

Error hierarchy
---------------

Base class
~~~~~~~~~~

.. autoclass:: genplanner.errors.GenPlannerBaseError
   :members:
   :show-inheritance:


Validation errors
~~~~~~~~~~~~~~~~~

.. autoclass:: genplanner.errors.SplitPolygonValidationError
   :members:
   :show-inheritance:

Raised when arguments passed to the internal polygon splitter
are invalid (wrong types, missing zones, malformed pairs, etc.).


Initialization errors
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: genplanner.errors.GenPlannerInitError
   :members:
   :show-inheritance:

Raised during :class:`~genplanner.GenPlanner` initialization
when territory input or preprocessing validation fails.


Argument errors
~~~~~~~~~~~~~~~

.. autoclass:: genplanner.errors.GenPlannerArgumentError
   :members:
   :show-inheritance:

Raised when invalid arguments are passed to public methods.

.. autoclass:: genplanner.errors.FixPointsOutsideTerritoryError
   :members:
   :show-inheritance:

Special case of ``GenPlannerArgumentError`` raised when fixed
anchor points lie outside the working territory.


Relation matrix errors
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: genplanner.errors.RelationMatrixError
   :members:
   :show-inheritance:

Raised when relation matrix construction or validation fails.


Optimization errors
~~~~~~~~~~~~~~~~~~~

.. autoclass:: genplanner.errors.GenplannerInfeasibleMultiFeatureError
   :members:
   :show-inheritance:

Raised when a multi-feature zoning problem cannot be solved
after repeated attempts.