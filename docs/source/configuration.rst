Configuration
=============

The module-level ``genplanner.config`` object holds defaults shared by the
planner's preprocessing steps. Change these before creating a planner when an
application needs different defaults.

``roads_width_def``
   Default road widths in meters, keyed by road level. The initial values are
   20 for ``high speed highway``, 10 for ``regulated highway``, and 5 for
   ``local road``.

``point_pool_seed`` and ``point_pool_size``
   Seed and size of the candidate point pool used during splitting. Their
   initial values are 42 and 200. The pool is created when the splitter module
   is imported, so changing these values later does not rebuild it.

``minimum_block_area``
   Default minimum area used by block generation, initially 20,000 square
   meters in the projected working CRS.

``change_logger_lvl(level)``
   Replace the Loguru sink on stderr. Accepted levels are ``TRACE``, ``DEBUG``,
   ``INFO``, ``WARN``, and ``ERROR``.

Example::

   from genplanner import config

   config.change_logger_lvl("DEBUG")

Planner-specific options such as ``parallel``, ``roads_extend_distance``, and
``simplify_geometry_value`` are arguments to :class:`genplanner.GenPlanner`.
Pass ``seed=7`` to ``GenPlanner`` to repeat Voronoi site placement with the same
inputs and runtime settings. Without ``seed``, each split uses a fresh random
seed.
``max_optimization_iterations`` limits each Voronoi optimization attempt
(default: 2000). ``max_run_seconds`` stops generation after 900 seconds by
default and raises ``TimeoutError`` with the number of completed tasks. Pass
``None`` to disable this limit. The time budget starts after input
preprocessing and does not return partial zones.
