.. image:: _static/logo_full.png
   :alt: GenPlanner logo
   :align: center

------------------------------------------------

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black

.. image:: https://img.shields.io/pypi/v/genplanner.svg
   :target: https://pypi.org/project/genplanner/

.. image:: https://github.com/DDonnyy/GenPlanner-lib/actions/workflows/release.yml/badge.svg
   :target: https://github.com/DDonnyy/GenPlanner-lib/actions/workflows/release.yml

.. image:: https://github.com/DDonnyy/GenPlanner-lib/actions/workflows/docs.yml/badge.svg
   :target: https://github.com/DDonnyy/GenPlanner-lib/actions/workflows/docs.yml

.. image:: https://img.shields.io/badge/GitHub-GenPlanner--lib-181717?logo=github
   :target: https://github.com/DDonnyy/GenPlanner-lib

.. image:: https://img.shields.io/badge/license-Proprietary--NC-red.svg
   :target: LICENSE

------------------------------------------------

GenPlanner is a territorial zoning engine for generating spatially consistent
territorial zones (and optionally blocks) from polygonal territories.

It combines:

- ratio-based functional zoning
- adjacency constraints via relation matrices
- optional fixed spatial anchors
- integration of existing zones
- road-based splitting
- MILP-based multi-feature allocation
- geometric optimization via Voronoi splitting

------------------------------------------------

Quickstart
----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install genplanner


Minimal example
~~~~~~~~~~~~~~~

.. code-block:: python

   import geopandas as gpd
   from genplanner import GenPlanner
   from genplanner.zones import basic_func_zone

   territory = gpd.read_file("territory.geojson")

   gp = GenPlanner(features_gdf=territory)

   zones, roads = gp.features2terr_zones(
       funczone=basic_func_zone
   )


Generate blocks
~~~~~~~~~~~~~~~

.. code-block:: python

   blocks, roads = gp.features2terr_zones2blocks(
       funczone=basic_func_zone
   )


With relation constraints
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   zones, roads = gp.features2terr_zones(
       relation_matrix="default"
   )


With fixed anchor points
~~~~~~~~~~~~~~~~~~~~~~~~

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

------------------------------------------------

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: Guide

   genplanner
   zones
   relations
   errors

------------------------------------------------

Design philosophy
-----------------

GenPlanner is built around a small set of composable primitives:

- :doc:`zones` define *what* spatial program should be achieved.
- :doc:`relations` define *which zones may or may not be adjacent*.
- :doc:`genplanner` orchestrates preprocessing, validation, optimization and splitting.
- :doc:`errors` define a strict, explicit failure model.

The goal is deterministic, reproducible zoning with clear constraints
and explicit spatial control.

License
-------

This project is distributed under a **Proprietary Non-Commercial License**.

You are allowed to use the software for non-commercial purposes only.

The following actions are prohibited without prior permission from the author:

- copying the source code
- modifying the software
- redistributing the software
- commercial use

See the full license text here:

- LICENSE_

.. _LICENSE: https://github.com/DDonnyy/GenPlanner-lib/blob/main/LICENSE
