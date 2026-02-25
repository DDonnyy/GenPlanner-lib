GenPlanner
==========

:class:`GenPlanner` is the main orchestration class of the library.
It prepares input territory geometry, applies preprocessing (roads, exclusions,
existing zones), and executes the zoning pipeline.

The planner operates in a projected local CRS (UTM), but all outputs are
returned in the original CRS of the input territory.

See also:

- :doc:`zones`
- :doc:`relations`

------------------------------------------------

Conceptual pipeline
-------------------

High-level workflow:

1. Prepare territory geometry.
2. Apply exclusions (optional).
3. Split by roads (optional).
4. Integrate existing territorial zones (optional).
5. Balance target ratios.
6. Build relation matrix.
7. Run territorial optimization.
8. (Optional) Split zones into blocks.

The heavy geometric split is performed by the internal
``split_polygon`` engine.

------------------------------------------------

Minimal example
---------------

.. code-block:: python

   import geopandas as gpd
   from genplanner import GenPlanner
   from genplanner.zones import basic_func_zone

   territory = gpd.read_file("territory.geojson")

   gp = GenPlanner(features_gdf=territory)

   zones_gdf, roads_gdf = gp.features2terr_zones(
       funczone=basic_func_zone
   )

Generate blocks instead of zone-level polygons:

.. code-block:: python

   blocks_gdf, roads_gdf = gp.features2terr_zones2blocks(
       funczone=basic_func_zone
   )

------------------------------------------------

Relation matrix
---------------

The ``relation_matrix`` argument controls adjacency rules between zones.
See :doc:`relations` for details.

Supported values:

- ``"default"`` — built from predefined forbidden neighborhood rules
- ``"empty"`` — no adjacency constraints
- :class:`~genplanner.ZoneRelationMatrix` instance
- ``None`` — treated as ``"default"``

Example:

.. code-block:: python

   from genplanner import ZoneRelationMatrix
   from genplanner.zones import default_terr_zones

   res = default_terr_zones.residential_terr
   ind = default_terr_zones.industrial_terr

   mat = ZoneRelationMatrix.from_pairs(
       zones=[res, ind],
       forbidden=[(res, ind)],
   )

   zones, roads = gp.features2terr_zones(
       relation_matrix=mat
   )

------------------------------------------------

Fixed zone anchor points
------------------------

Both zoning methods accept an optional argument:

``terr_zones_fix_points: GeoDataFrame``

It allows you to constrain spatial allocation by specifying
anchor points for particular territorial zones.

Each row represents a fixed spatial hint for a zone.

Requirements:

- Geometry must be ``Point``
- Column ``fixed_zone`` must be present
- ``fixed_zone`` values must be :class:`~genplanner.zones.TerritoryZone` objects
  present in ``funczone.zones_ratio``

All fixed points must lie inside the working territory.

Example:

.. code-block:: python

   import geopandas as gpd
   from shapely.geometry import Point
   from genplanner.zones import default_terr_zones

   fix_points = gpd.GeoDataFrame(
       {
           "fixed_zone": [
               default_terr_zones.residential_terr,
               default_terr_zones.business_terr,
           ],
           "geometry": [
               Point(30.1, 59.9),
               Point(30.2, 59.91),
           ],
       },
       crs="EPSG:4326",
   )

   zones, roads = gp.features2terr_zones(
       terr_zones_fix_points=fix_points
   )

------------------------------------------------

Existing zones integration
--------------------------

If ``existing_terr_zones`` is provided:

- Territory fragments sufficiently covered by existing zones are merged.
- Static fix points are derived automatically.
- Target ratios are rebalanced.
- Final output always includes existing zones.

See internal validation logic for details.

------------------------------------------------

Full example with roads, exclusions, existing zones and fixed anchors
----------------------------------------------------------------------

.. code-block:: python

   import geopandas as gpd
   from shapely.geometry import Point
   from genplanner import GenPlanner, default_terr_zones

   roads = gpd.read_file("roads.geojson")
   territory = gpd.read_file("territory.geojson")
   exclude = gpd.read_file("exclude_features.geojson")
   existing_zones = gpd.read_file("existing_zones.geojson")

   terr_zone_map = {
       "industrial": default_terr_zones.industrial_terr,
       "special": default_terr_zones.special_terr,
   }

   existing_zones["territory_zone"] = (
       existing_zones["territory_zone"]
       .map(terr_zone_map)
   )

   # Optional: add anchor points for certain zones
   fix_points = gpd.GeoDataFrame(
       {
           "fixed_zone": [
               default_terr_zones.industrial_terr,
           ],
           "geometry": [
               Point(30.15, 59.92),
           ],
       },
       crs=territory.crs,
   )

   gp = GenPlanner(
       features_gdf=territory,
       roads_gdf=roads,
       exclude_gdf=exclude,
       existing_terr_zones=existing_zones,
       parallel=True,
   )

   new_zones, new_roads = gp.features2terr_zones2blocks(
       terr_zones_fix_points=fix_points
   )

Visualize:

.. code-block:: python

   m = new_zones.explore(column="territory_zone", tiles="cartodb positron")
   new_roads.explore(m=m, column="road_lvl")

------------------------------------------------

Parallel execution
------------------

If ``parallel=True`` (default), the planner uses
``ProcessPoolExecutor`` for downstream tasks.

- Worker count defaults to ``cpu_count - 1``.
- Falls back to single-thread mode if only one CPU is available.

------------------------------------------------

Logging
-------

If ``rust_write_logs=True``:

- A directory named after ``run_name`` is created.
- Optimizer logs are written per split attempt.

Useful for debugging zoning instability.

------------------------------------------------

Returned objects
----------------

Both zoning methods return:

``(zones_or_blocks_gdf, roads_gdf)``

- All geometries are returned in the original input CRS.
- ``roads_width`` column is always present.
- ``road_lvl`` column describes road type.

------------------------------------------------

API reference
-------------

.. currentmodule:: genplanner

.. autosummary::
   :toctree: generated
   :nosignatures:

   GenPlanner