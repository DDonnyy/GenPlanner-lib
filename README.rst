.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black

.. image:: https://img.shields.io/pypi/v/genplanner.svg
   :target: https://pypi.org/project/genplanner/

.. image:: https://github.com/DDonnyy/GenPlanner-lib/actions/workflows/release.yml/badge.svg
   :target: https://github.com/DDonnyy/GenPlanner-lib/actions/workflows/release.yml

.. image:: https://github.com/DDonnyy/GenPlanner-lib/actions/workflows/docs.yml/badge.svg
   :target: https://github.com/DDonnyy/GenPlanner-lib/actions/workflows/docs.yml

.. image:: https://img.shields.io/badge/docs-GitHub%20Pages-blue
   :target: https://ddonnyy.github.io/GenPlanner-lib/

.. image:: https://img.shields.io/badge/license-BSD--3--Clause-blue.svg
   :target: https://opensource.org/licenses/BSD-3-Clause

------------------------------------------------

.. image:: docs/source/_static/logo_full.png
   :alt: GenPlanner logo
   :align: center
   :width: 600px

------------------------------------------------

GenPlanner is a territorial zoning engine for generating spatially consistent
territorial zones (and optionally blocks) from polygonal territories.

It combines:

- ratio-based functional zoning
- adjacency constraints via relation matrices
- optional spatial anchor points
- integration of existing zones
- road-based splitting
- MILP-based multi-feature allocation
- geometric optimization via Voronoi splitting


Installation
------------

.. code-block:: bash

   pip install genplanner


Quick Example
-------------

.. code-block:: python

   import geopandas as gpd
   from genplanner import GenPlanner
   from genplanner.zones import basic_func_zone

   territory = gpd.read_file("territory.geojson")

   gp = GenPlanner(features_gdf=territory)

   zones, roads = gp.features2terr_zones(
       funczone=basic_func_zone
   )

Generate blocks:

.. code-block:: python

   blocks, roads = gp.features2terr_zones2blocks(
       funczone=basic_func_zone
   )


Example with relation constraints
---------------------------------

.. code-block:: python

   zones, roads = gp.features2terr_zones(
       relation_matrix="default"
   )


Example with spatial anchors
----------------------------

.. code-block:: python

   from shapely.geometry import Point
   from genplanner.zones import default_terr_zones

   fix_points = gpd.GeoDataFrame(
       {
           "fixed_zone": [default_terr_zones.residential_terr],
           "geometry": [Point(30.1, 59.9)],
       },
       crs=territory.crs,
   )

   zones, roads = gp.features2terr_zones(
       terr_zones_fix_points=fix_points
   )


Documentation
-------------

Full documentation is available at:

https://ddonnyy.github.io/GenPlanner-lib/

Core modules:

- GenPlanner
- Zones
- Zone Relations
- Errors


Design Philosophy
-----------------

GenPlanner is built around a small set of composable primitives:

- Zones define *what* spatial program should be achieved.
- Relations define *which zones may be adjacent*.
- The planner orchestrates preprocessing, validation and optimization.
- Errors are explicit and structured.

The goal is deterministic, reproducible zoning with clear constraints
and explicit spatial control.