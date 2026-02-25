Zone Relations
==============

Zone relations define adjacency constraints between zones.
They are represented as a symmetric matrix where each cell specifies
how two zones are allowed (or forbidden) to neighbor each other.

Relations are optional. If not provided, GenPlanner uses a default matrix
built from forbidden neighborhood rules.

Relation values
---------------

The matrix cells use the :class:`Relation` enum:

- ``NEUTRAL`` — no constraint
- ``NEIGHBOR`` — zones must / are encouraged to be neighbors
- ``FORBIDDEN`` — zones cannot be neighbors

Example:

.. code-block:: python

   from genplanner import Relation

   print(Relation.FORBIDDEN.value)  # "forbidden"

------------------------------------------------

ZoneRelationMatrix
------------------

:class:`ZoneRelationMatrix` is a symmetric adjacency matrix with O(1) access.

Key properties:

- Matrix is always symmetric.
- Default relation is ``NEUTRAL``.
- Zones must be unique and hashable.
- Works with any :class:`Zone`, but typically used with
  :class:`~genplanner.zones.TerritoryZone`.

Build Empty matrix
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from genplanner import ZoneRelationMatrix
   from genplanner.zones import default_terr_zones

   zones = [
       default_terr_zones.residential_terr,
       default_terr_zones.industrial_terr,
   ]

   mat = ZoneRelationMatrix.empty(zones)

   print(mat.is_forbidden(zones[0], zones[1]))  # False


Build from explicit pairs
~~~~~~~~~~~~~~~~~~~~~~~~~

You can explicitly define neighbor or forbidden pairs:

.. code-block:: python

   from genplanner import ZoneRelationMatrix
   from genplanner.zones import default_terr_zones

   res = default_terr_zones.residential_terr
   ind = default_terr_zones.industrial_terr

   mat = ZoneRelationMatrix.from_pairs(
       zones=[res, ind],
       forbidden=[(res, ind)]
   )

   assert mat.is_forbidden(res, ind)


Build from_kind_forbidden (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GenPlanner ships with a predefined set:

.. code-block:: python

   from genplanner import FORBIDDEN_NEIGHBORHOOD

   print(FORBIDDEN_NEIGHBORHOOD)

The set contains pairs of :class:`TerritoryZoneKind` that must not neighbor.

How it works
^^^^^^^^^^^^

``from_kind_forbidden``:

1. Groups zones by their ``kind``.
2. For each forbidden kind pair ``(A, B)``,
   marks **all zones of kind A** and **all zones of kind B**
   as mutually forbidden.
3. Ensures symmetry.

Example
^^^^^^^

.. code-block:: python

   from genplanner import ZoneRelationMatrix, FORBIDDEN_NEIGHBORHOOD
   from genplanner.zones import default_terr_zones

   zones = [
       default_terr_zones.residential_terr,
       default_terr_zones.industrial_terr,
       default_terr_zones.business_terr,
   ]

   mat = ZoneRelationMatrix.from_kind_forbidden(
       zones=zones,
       kind_forbidden=FORBIDDEN_NEIGHBORHOOD,
   )

   # residential vs industrial is forbidden (by default rules)
   assert mat.is_forbidden(
       default_terr_zones.residential_terr,
       default_terr_zones.industrial_terr
   )



Build from DataFrame
~~~~~~~~~~~~~~~~~~~~

You can construct a matrix from a square pandas DataFrame.

Requirements:

- ``df.index`` contains :class:`~genplanner.zones.abc_zone.Zone` objects
- ``df.columns`` contains :class:`~genplanner.zones.abc_zone.Zone` objects
- ``set(df.index) == set(df.columns)``

Example (Zone objects as labels)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import pandas as pd
   from genplanner import ZoneRelationMatrix
   from genplanner.zones import default_terr_zones

   res = default_terr_zones.residential_terr
   ind = default_terr_zones.industrial_terr

   df = pd.DataFrame(
       data=[
           [0,  -1],
           [-1,  0],
       ],
       index=[res, ind],
       columns=[res, ind],
   )

   mat = ZoneRelationMatrix.from_dataframe(df)

   assert mat.is_forbidden(res, ind)

Allow non-symmetric input
^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to load a DataFrame that is not symmetric (e.g., partially filled),
disable symmetry validation:

.. code-block:: python

   mat = ZoneRelationMatrix.from_dataframe(df, require_symmetric=False)


Matrix Subsets
~~~~~~~~~~~~~~

You can restrict a matrix to a subset of zones:

.. code-block:: python

   sub = mat.subset([res])
   print(sub.zones)



Pretty print
~~~~~~~~~~~~

The matrix can be visualized:

.. code-block:: python

   print(mat.to_pretty_string())

Output example:

::

         res   ind   bus
   res    ·     ✗     ·
   ind    ✗     ·     ·
   bus    ·     ·     ·

------------------------------------------------

Using with GenPlanner
---------------------

Pass a matrix to ``features2terr_zones``:

.. code-block:: python

   from genplanner import GenPlanner

   gp = GenPlanner(features_gdf=features)

   zones, roads = gp.features2terr_zones(
       funczone=basic_func_zone,
       relation_matrix=mat,
   )


------------------------------------------------

API reference
-------------

.. currentmodule:: genplanner

Core classes
~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:

   ZoneRelationMatrix
   Relation
   FORBIDDEN_NEIGHBORHOOD