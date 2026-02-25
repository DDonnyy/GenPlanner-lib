Zoning Primitives
=================

This page describes the zone objects used by GenPlanner to define zoning goals,
constraints, and presets. Zones are lightweight Python objects (hashable and
orderable) that are used as keys in ratio mappings and as identifiers across
the pipeline.

Overview
--------

GenPlanner uses four main zone classes:

- ``Zone``: abstract base interface for all zones.
- ``BasicZone``: minimal zone identified only by its name.
- ``TerritoryZone``: a concrete territorial zone type (kind + name + min block area).
- ``FunctionalZone``: a composite zone describing the desired mix of territorial zones.

.. note::

   ``GenPlan`` and ``gen_plan`` exist in the codebase but are intentionally not
   documented here.

------------------------------------------------

Zone (abstract base)
--------------------

``Zone`` defines the minimal interface and shared behavior for all zone types:

- ``name``: a human-readable identifier.
- ``min_area``: generic minimum area constraint (defaults to ``1.0``).
- hash + ordering: zones are comparable and hashable by ``(name, min_area)``.

Example: ordering and hashing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from genplanner.zones import Zone, BasicZone

   a = BasicZone("block_a")
   b = BasicZone("block_b")

   assert a != b
   assert a < b  # ordering by (name, min_area)
   s = {a, b}    # hashable -> can be used in sets / dict keys

------------------------------------------------

BasicZone
---------

``BasicZone`` is a minimal concrete implementation of ``Zone`` that only stores
a non-empty string name. It is typically used for technical or intermediate
concepts (e.g., generated blocks).

Example
~~~~~~~

.. code-block:: python

   from genplanner.zones import BasicZone

   z = BasicZone("block_1")
   print(z)  # Zone "block_1"

------------------------------------------------

TerritoryZoneKind
-----------------

``TerritoryZoneKind`` is an enum of supported territorial zone kinds. Its values
are stable string identifiers used in outputs and configuration.

Example
~~~~~~~

.. code-block:: python

   from genplanner.zones.territory_zones import TerritoryZoneKind

   assert TerritoryZoneKind.RESIDENTIAL.value == "residential"

Relation matrix and kind-based rules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Territorial zone kinds are also used to derive adjacency constraints.
See :doc:`relations` for details on how forbidden neighborhood rules
are converted into a symmetric relation matrix using
``ZoneRelationMatrix.from_kind_forbidden``.

TerritoryZone
-------------

``TerritoryZone`` is the core unit of territorial zoning. It defines:

- ``kind``: one of :class:`~genplanner.zones.territory_zones.TerritoryZoneKind`
- ``name``: a non-empty name used as an identifier in outputs and in fixed points
- ``min_block_area``: a positive number controlling minimum block size for that zone

For compatibility with generic constraints, ``TerritoryZone.min_area`` is defined
as ``min_block_area``. ``kind`` can be provided as an enum or a value convertible
to it (it is coerced on validation).

Example: custom territorial zone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from genplanner.zones import TerritoryZone
   from genplanner.zones.territory_zones import TerritoryZoneKind

   tech_park = TerritoryZone(
       kind=TerritoryZoneKind.BUSINESS,
       name="tech_park",
       min_block_area=50_000,
   )

   print(tech_park)  # Territory zone "tech_park" (business)

Example: coercing kind from string
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from genplanner.zones import TerritoryZone

   # kind may be coerced from string values
   z = TerritoryZone(kind="residential", name="res", min_block_area=10_000)
   assert z.kind.value == "residential"

------------------------------------------------

FunctionalZone
--------------

``FunctionalZone`` describes a *program* (mix) of territorial zones using target
area ratios. It stores a mapping:

``{TerritoryZone -> ratio}``

Validation rules:

- ``name`` must be a non-empty string
- mapping must be non-empty
- keys must be ``TerritoryZone``
- ratios must be finite numbers > 0

Ratios are normalized to sum to ``1.0`` on initialization. The functional zone's
``min_area`` is computed as the weighted sum:

``sum(zone.min_area * ratio)`` using the normalized ratios.

Example: define a functional program
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from genplanner.zones import FunctionalZone
   from genplanner.zones.territory_zones import (
       residential_terr,
       business_terr,
       recreation_terr,
   )

   downtown = FunctionalZone(
       zones_ratio={
           residential_terr: 0.6,
           business_terr: 0.3,
           recreation_terr: 0.1,
       },
       name="downtown",
   )

   # ratios are normalized internally
   ratios = downtown.zones_ratio
   assert abs(sum(ratios.values()) - 1.0) < 1e-9

------------------------------------------------

Preset zones
------------

GenPlanner ships with a set of predefined territorial zones and functional presets.

Basic functional zone
~~~~~~~~~~~~~~~~~~~~~

``basic_func_zone`` is a balanced preset that defines a default target mix across
all default territorial zones.

.. code-block:: python

   from genplanner.zones import basic_func_zone

   print(basic_func_zone.name)          # "basic"
   print(basic_func_zone.zones_ratio)   # mapping {TerritoryZone -> ratio}

Default territorial zones
~~~~~~~~~~~~~~~~~~~~~~~~~

``default_terr_zones`` is a module-like collection of predefined territorial zones:

- ``residential_terr``
- ``industrial_terr``
- ``business_terr``
- ``recreation_terr``
- ``transport_terr``
- ``agriculture_terr``
- ``special_terr``

.. code-block:: python

   from genplanner.zones import default_terr_zones

   # Access as attributes on the module
   print(default_terr_zones.residential_terr)
   print(default_terr_zones.industrial_terr.min_block_area)

Default functional presets
~~~~~~~~~~~~~~~~~~~~~~~~~~

``default_func_zones`` is a module-like collection of predefined functional programs:

- ``residential_func_zone``
- ``industrial_func_zone``
- ``business_func_zone``
- ``recreation_func_zone``
- ``transport_func_zone``
- ``agricalture_func_zone``
- ``special_func_zone``

.. code-block:: python

   from genplanner.zones import default_func_zones

   fz = default_func_zones.residential_func_zone
   print(fz.name)
   print(sum(fz.zones_ratio.values()))  # always 1.0 after normalization

------------------------------------------------

Using zones with GenPlanner
---------------------------

In most cases you only need to pass a functional zone preset into the planner:

.. code-block:: python

   import geopandas as gpd
   from genplanner import GenPlanner
   from genplanner.zones import basic_func_zone

   features = gpd.read_file("territory.geojson")

   gp = GenPlanner(features_gdf=features)
   zones_gdf, roads_gdf = gp.features2terr_zones(funczone=basic_func_zone)

.. tip::
   In fixed points GeoDataFrame, the ``fixed_zone`` column is matched against
   the keys of ``funczone.zones_ratio``. In the current implementation those keys
   are :class:`~genplanner.zones.TerritoryZone` objects, so ``fixed_zone`` should
   contain the corresponding ``TerritoryZone`` instances.

.. code-block:: python

   import geopandas as gpd
   from shapely.geometry import Point
   from genplanner.zones import default_terr_zones

   fix_points = gpd.GeoDataFrame(
       {
           "fixed_zone": [default_terr_zones.residential_terr],
           "geometry": [Point(10, 10)],
       },
       crs="EPSG:4326",
   )

------------------------------------------------

API reference
-------------

.. currentmodule:: genplanner

Core zone classes
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:

   Zone
   BasicZone
   TerritoryZone
   FunctionalZone


Zone presets
~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:

   basic_func_zone
   default_func_zones
   default_terr_zones


Territorial zone kinds
~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: genplanner.zones.territory_zones

.. autosummary::
   :toctree: generated
   :nosignatures:

   TerritoryZoneKind