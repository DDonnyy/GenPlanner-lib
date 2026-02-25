import concurrent.futures
import multiprocessing
import os
import queue
from datetime import datetime
from pathlib import Path
from typing import Literal

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon

from genplanner import config
from genplanner.errors import GenPlannerArgumentError, GenPlannerInitError
from genplanner.main.init_validation import (
    add_static_fix_points,
    cut_by_existing_terr_zones,
    cut_by_roads,
    cut_out_features,
    prepare_fixed_points_and_balance_ratios,
    resolve_relation_matrix,
)
from genplanner.tasks import (
    feature2terr_zones_initial,
    multi_feature2terr_zones_initial,
)
from genplanner.utils import territory_splitter
from genplanner.zone_relations.relation_matrix import ZoneRelationMatrix
from genplanner.zones import FunctionalZone, basic_func_zone

logger = config.logger


class GenPlanner:
    """
    Attributes:
        original_territory: Copy of the input `features_gdf`.
        original_crs: CRS of the input territory.
        local_crs: Estimated local projected CRS (UTM).
        territory_to_work_with: Prepared working territory polygons in `local_crs`.
        existing_terr_zones: Prepared existing zones in `local_crs`.
        static_fix_points: Fixed points derived from existing zones (column `fixed_zone`).
        user_roads: Prepared roads in `local_crs` (column `roads_width` always present).
        source_multipolygon: True if the working territory contains multiple polygons.
        run_name: Run identifier (and log directory prefix).
    """

    def __init__(
        self,
        features_gdf: gpd.GeoDataFrame,
        roads_gdf: gpd.GeoDataFrame | None = None,
        roads_extend_distance: float = 5,
        exclude_gdf: gpd.GeoDataFrame | None = None,
        exclude_buffer: float = 5,
        existing_terr_zones: gpd.GeoDataFrame | None = None,
        existing_tz_fill_ratio: float = 0.7,
        existing_tz_merge_radius=50,
        simplify_geometry_value=0.01,
        parallel=True,
        parallel_max_workers=None,
        rust_write_logs=False,
        run_name=None,
    ):
        """
        Initialize a territory zoning pipeline.

        The planner prepares the input territory geometry into a working GeoDataFrame
        in a local projected CRS, optionally applies exclusions, cuts the territory by
        roads, integrates existing territorial zones, and configures runtime settings
        such as parallel execution and optimizer logging.

        High-level preprocessing steps:
          1) Keep only Polygon/MultiPolygon features from `features_gdf`.
          2) Reproject to a local CRS (estimated UTM) and explode multipart geometries.
          3) If `exclude_gdf` is provided: clip, buffer by `exclude_buffer`, subtract.
          4) If `roads_gdf` is provided: normalize/deduplicate, extend by
             `roads_extend_distance`, split the territory, and keep roads that intersect
             produced splitters; fill missing `roads_width` with a default and warn.
          5) If `existing_terr_zones` is provided: validate schema, clip to territory,
             merge territory fragments into existing zones when coverage ratio exceeds
             `existing_tz_fill_ratio`, then cut the remaining territory by those zones.
          6) If existing zones exist: build static fixed points from existing zones
             using `existing_tz_merge_radius` and snap them to the territory boundary.

        Args:
            features_gdf:
                Input territory geometry as a GeoDataFrame. Only Polygon and MultiPolygon
                geometries are used. If the final working set has more than one polygon,
                the planner will run a "multi feature" task variant.
            roads_gdf:
                Optional roads GeoDataFrame used to split the territory. Any geometry type
                is accepted, but it is expected to contain line-like geometries after
                normalization. The `roads_width` column is optional; if missing, it will
                be filled with a default width and a warning will be logged.
            roads_extend_distance:
                Distance (in local CRS units, typically meters) used to extend road
                linestrings before splitting.
            exclude_gdf:
                Optional exclusion geometries (e.g., water, protected areas). If provided,
                they are buffered and subtracted from the territory.
            exclude_buffer:
                Buffer distance (local CRS units) applied to `exclude_gdf` before subtraction.
            existing_terr_zones:
                Optional existing territorial zones to preserve and merge into the output.
                If provided, it must contain a `territory_zone` column.
            existing_tz_fill_ratio:
                Minimum ratio of a territory polygon area that must be covered by an
                existing zone for that polygon to be merged into the existing zone
                instead of being split further. Must be in [0, 1].
            existing_tz_merge_radius:
                Radius (local CRS units) used to merge nearby fragments of the same
                existing zone when deriving static fix points, and to stabilize representative
                point selection.
            simplify_geometry_value:
                Internal simplification factor in normalized 0..1 space used for speed/robustness
                trade-offs. It does not directly change the final output geometry, but can
                influence processing.
            parallel:
                Whether to use multiprocessing for downstream tasks. If disabled or if the
                machine has fewer than 2 CPUs, execution is forced to single-worker mode.
            parallel_max_workers:
                Maximum worker count for parallel execution. If None and parallel is enabled,
                defaults to `max(1, cpu_count - 1)`.
            rust_write_logs:
                If True, enables Rust optimizer logs and creates a directory for `run_name`.
            run_name:
                Optional run identifier used as a log/artifact prefix. If None, an auto name
                like `gp_DDMMYY_HH_MM` is generated.

        Raises:
            GenPlannerInitError:
                If `features_gdf` has no valid Polygon/MultiPolygon geometries, or CRS checks fail,
                or `existing_terr_zones` is provided without the required `territory_zone` column.

        """
        self.original_territory = features_gdf.copy()
        self.original_crs = features_gdf.crs
        self.local_crs = features_gdf.estimate_utm_crs()
        self.simplify_geometry_value = simplify_geometry_value
        self.existing_terr_zones = gpd.GeoDataFrame()
        self.static_fix_points = gpd.GeoDataFrame()
        self.territory_to_work_with = gpd.GeoDataFrame()

        if roads_gdf is None:
            roads_gdf = gpd.GeoDataFrame()
        if exclude_gdf is None:
            exclude_gdf = gpd.GeoDataFrame()
        if existing_terr_zones is None:
            existing_terr_zones = gpd.GeoDataFrame()

        self._create_working_gdf(
            features_gdf=features_gdf,
            exclude_gdf=exclude_gdf,
            exclude_buffer=exclude_buffer,
            roads_gdf=roads_gdf,
            roads_extend_distance=roads_extend_distance,
            existing_terr_zones=existing_terr_zones,
            existing_tz_fill_ratio=existing_tz_fill_ratio,
            existing_tz_merge_radius=existing_tz_merge_radius,
        )

        self.parallel = bool(parallel)
        self.rust_write_logs = bool(rust_write_logs)

        cpu_total = os.cpu_count() or 1

        if not self.parallel or cpu_total < 2:
            self.parallel_max_workers = 1
            logger.info("Parallel execution disabled (ProcessPoolExecutor off).")
        else:
            if parallel_max_workers is None:
                self.parallel_max_workers = max(1, cpu_total - 1)
                logger.debug(f"Parallel execution enabled | workers={self.parallel_max_workers} (auto)")
            else:
                self.parallel_max_workers = max(1, int(parallel_max_workers))
                logger.debug(f"Parallel execution enabled | workers={self.parallel_max_workers} (manual)")
        if self.rust_write_logs:
            logger.info("Rust optimizer logs enabled.")
        if run_name is None:
            now = datetime.now()
            self.run_name = now.strftime("gp_%d%m%y_%H_%M")
        else:
            self.run_name = str(run_name)

    def _create_working_gdf(
        self,
        features_gdf: gpd.GeoDataFrame,
        exclude_gdf: gpd.GeoDataFrame,
        exclude_buffer: float,
        roads_gdf: gpd.GeoDataFrame,
        roads_extend_distance: float,
        existing_terr_zones: gpd.GeoDataFrame,
        existing_tz_fill_ratio: float,
        existing_tz_merge_radius: float,
    ):

        features_gdf = features_gdf[features_gdf.geom_type.isin(["MultiPolygon", "Polygon"])]

        if len(features_gdf) == 0:
            raise GenPlannerInitError("No valid geometries in provided territory_gdf GeoDataFrame")

        features_gdf = features_gdf.to_crs(self.local_crs)
        features_gdf = features_gdf.explode(ignore_index=True)

        if len(exclude_gdf) > 0:
            exclude_gdf = exclude_gdf.to_crs(self.local_crs)
            features_gdf = cut_out_features(features_gdf, exclude_gdf, exclude_buffer)

        if len(roads_gdf) > 0:
            roads_gdf = roads_gdf.to_crs(self.local_crs)
            features_gdf, roads_gdf = cut_by_roads(features_gdf, roads_gdf, roads_extend_distance)
            self.user_roads = roads_gdf
        else:
            self.user_roads = gpd.GeoDataFrame()

        if len(existing_terr_zones) > 0:
            existing_terr_zones = existing_terr_zones.to_crs(self.local_crs)
            features_gdf, existing_terr_zones = cut_by_existing_terr_zones(
                features_gdf, existing_terr_zones, existing_tz_fill_ratio
            )
            self.existing_terr_zones = existing_terr_zones

        if len(existing_terr_zones) > 0:
            self.static_fix_points = add_static_fix_points(features_gdf, existing_terr_zones, existing_tz_merge_radius)

        self.source_multipolygon = not len(features_gdf) == 1
        self.territory_to_work_with = features_gdf

    def _run(self, initial_func, *args, **kwargs):
        task_queue = multiprocessing.Queue()
        run_name = f"{self.run_name}/"
        kwargs.update(
            {
                "rust_write_logs": self.rust_write_logs,
                "simplify": self.simplify_geometry_value,
                "local_crs": self.local_crs.to_epsg(),
                "run_name": run_name,
            }
        )
        if self.rust_write_logs:
            log_path = Path(run_name)
            log_path.mkdir(parents=True, exist_ok=True)
        task_queue.put((initial_func, args, kwargs))
        generated_zones, generated_roads = split_queue(
            task_queue, self.local_crs, parallel=self.parallel, max_workers=self.parallel_max_workers
        )

        complete_zones = pd.concat([generated_zones, self.existing_terr_zones], ignore_index=True)
        generated_roads = pd.concat([generated_roads, self.user_roads], ignore_index=True)

        roads_poly = generated_roads.copy()
        roads_poly.geometry = roads_poly.apply(lambda x: x.geometry.buffer(x.roads_width / 2, resolution=4), axis=1)

        complete_zones = territory_splitter(complete_zones, roads_poly, reproject_attr=True).reset_index(drop=True)

        return complete_zones.to_crs(self.original_crs), generated_roads.to_crs(self.original_crs)

    RelationMatrixArg = ZoneRelationMatrix | Literal["empty", "default"] | None

    def features2terr_zones(
        self,
        funczone: FunctionalZone = basic_func_zone,
        relation_matrix: RelationMatrixArg = "default",
        terr_zones_fix_points: gpd.GeoDataFrame = None,
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Generate territorial zones for the prepared territory.

        Runs the full zoning pipeline and produces territorial zone polygons
        without further subdivision into blocks. The method respects existing
        territorial zones (if provided at initialization), integrates static
        fix points derived from them, and balances target ratios accordingly.

        The pipeline performs:
          - validation and optional adjustment of zone ratios,
          - validation and merging of fixed points,
          - resolution of the relation matrix (adjacency/forbidden rules),
          - task execution (single or multi-feature),
          - merging of generated and existing zones,
          - road-based splitting of final geometries.

        Args:
            funczone:
                Functional zoning definition containing a mapping
                of territorial zone kinds to target area ratios.
                Must be an instance of ``FunctionalZone``.

            relation_matrix:
                Zone adjacency/preference definition.

                Supported values:
                  - "default": builds a matrix using predefined forbidden
                    neighborhood rules for the zone kinds in ``funczone``.
                  - "empty": creates an empty relation matrix with no
                    encoded adjacency constraints.
                  - ZoneRelationMatrix instance: validated against the
                    zone kinds of ``funczone`` (missing zones raise an error;
                    extra zones produce a warning).
                  - None: treated as "default".

            terr_zones_fix_points:
                Optional GeoDataFrame of fixed zone anchor points.

                Required schema:
                  - geometry: Point (all geometries must be Points)
                  - fixed_zone: str (must match keys of ``funczone.zones_ratio``)

                Validation rules:
                  - All geometries must be Points.
                  - All ``fixed_zone`` values must exist in the zone ratio dict.
                  - All points must lie within the working territory.
                  - If existing zones reduce a zone’s remaining ratio to zero,
                    corresponding fixed points are removed automatically.

        Returns:
            tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
                A pair ``(zones_gdf, roads_gdf)`` in the original input CRS:

                  - zones_gdf:
                        Final territorial zone polygons, including merged
                        existing zones (if provided).
                  - roads_gdf:
                        Road geometries used/generated during splitting.
                        The column ``roads_width`` is always present.

        Raises:
            GenPlannerArgumentError:
                If ``funczone`` is not a ``FunctionalZone`` or if
                ``terr_zones_fix_points`` schema is invalid.

            FixPointsOutsideTerritoryError:
                If any fixed point lies outside the working territory.

            RelationMatrixError:
                If the relation matrix is invalid or inconsistent with
                the zone kinds in ``funczone``.

        """
        return self._features2terr_zones(funczone, relation_matrix, terr_zones_fix_points, split_further=False)

    def features2terr_zones2blocks(
        self,
        funczone: FunctionalZone = basic_func_zone,
        relation_matrix: RelationMatrixArg = "default",
        terr_zones_fix_points: gpd.GeoDataFrame = None,
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Generate territorial zones and subdivide them into blocks.

        Executes the same zoning pipeline as :meth:`features2terr_zones`,
        but additionally performs subdivision of territorial zone polygons
        into smaller block polygons according to zone-specific constraints
        (e.g., minimum block area).

        All preprocessing steps, validation rules, relation matrix handling,
        fixed point integration, and existing zone balancing are identical
        to :meth:`features2terr_zones`.

        Args:
            funczone:
                Functional zoning definition containing territorial zone kinds
                and their target area ratios. Must be a ``FunctionalZone``.

            relation_matrix:
                Zone adjacency/preference definition. Supported values:

                  - "default": builds a relation matrix from predefined
                    forbidden neighborhood rules.
                  - "empty": creates an empty relation matrix.
                  - ZoneRelationMatrix instance: validated against
                    ``funczone`` zone kinds.
                  - None: treated as "default".

            terr_zones_fix_points:
                Optional GeoDataFrame of fixed zone anchor points.

                Required schema:
                  - geometry: Point
                  - fixed_zone: str (must match zone names in ``funczone``)

                All validation and balancing rules are identical to
                :meth:`features2terr_zones`.

        Returns:
            tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
                A pair ``(blocks_gdf, roads_gdf)`` in the original input CRS:

                  - blocks_gdf:
                        Subdivided polygons after the additional block-splitting
                        stage (may include both zone-level and block-level outputs,
                        depending on downstream logic).
                  - roads_gdf:
                        Road geometries used/generated during splitting.

        Raises:
            GenPlannerArgumentError:
                If ``funczone`` is invalid or fixed point schema is incorrect.

            FixPointsOutsideTerritoryError:
                If any fixed point lies outside the working territory.

            RelationMatrixError:
                If the relation matrix is invalid or inconsistent.

        """
        return self._features2terr_zones(funczone, relation_matrix, terr_zones_fix_points, split_further=True)

    def _features2terr_zones(
        self,
        funczone: FunctionalZone,
        relation_matrix: RelationMatrixArg,
        terr_zones_fix_points: gpd.GeoDataFrame,
        split_further: bool,
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:

        if not isinstance(funczone, FunctionalZone):
            raise GenPlannerArgumentError("funczone arg must be of type FunctionalZone")

        new_fixed_zones, new_zone_ratio = prepare_fixed_points_and_balance_ratios(
            funczone.zones_ratio,
            terr_zones_fix_points,
            self.static_fix_points,
            self.territory_to_work_with,
            self.existing_terr_zones,
        )
        new_funczone = FunctionalZone(new_zone_ratio, funczone.name)

        relation_matrix = resolve_relation_matrix(new_funczone, relation_matrix)

        args = self.territory_to_work_with, new_funczone, relation_matrix, new_fixed_zones, split_further

        if self.source_multipolygon:
            return self._run(multi_feature2terr_zones_initial, *args)
        return self._run(feature2terr_zones_initial, *args)

    # def split_features(
    #     self, zones_ratio_dict: dict = None, zones_n: int = None, roads_width=None, fixed_zones: gpd.GeoDataFrame = None
    # ):
    #     """
    #     Splits every feature in working gdf according to provided zones_ratio_dict or zones_n
    #     """
    #     if zones_ratio_dict is None and zones_n is None:
    #         raise RuntimeError("Either zones_ratio_dict or zones_n must be set")
    #     if zones_ratio_dict is not None and len(zones_ratio_dict) in [0, 1]:
    #         raise ValueError("zones_ratio_dict ")
    #     if fixed_zones is None:
    #         fixed_zones = gpd.GeoDataFrame()
    #     if len(fixed_zones) > 0:
    #         if zones_ratio_dict is None:
    #             raise ValueError("zones_ratio_dict should not be None for generating fixed zones")
    #         fixed_zones = self._prepare_fixed_zones_and_balance_ratios(zones_ratio_dict, fixed_zones)
    #     if zones_n is not None:
    #         zones_ratio_dict = {x: 1 / zones_n for x in range(zones_n)}
    #     if len(zones_ratio_dict) > 8:
    #         raise RuntimeError("Use poly2block, to split more than 8 parts")
    #     args = (self.territory_to_work_with, zones_ratio_dict, roads_width, fixed_zones)
    #     return self._run(gdf_splitter, *args)

    # def features2blocks(self, terr_zone: TerritoryZone) -> (gpd.GeoDataFrame, gpd.GeoDataFrame):
    #     if not isinstance(terr_zone, TerritoryZone):
    #         raise TypeError("terr_zone arg must be of type TerritoryZone")
    #     if not "territory_zone" in self.territory_to_work_with.columns:
    #         logger.warning(
    #             f"territory_zone column not found in working gdf. All geometry's territory zone set to {terr_zone}"
    #         )
    #         self.territory_to_work_with["territory_zone"] = terr_zone
    #     return self._run(multi_feature2blocks_initial, self.territory_to_work_with)


def _merge_gdfs(gdfs: list[gpd.GeoDataFrame], local_crs):
    if not gdfs:
        return gpd.GeoDataFrame()
    return gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=local_crs, geometry="geometry")


def split_queue(
    task_queue: multiprocessing.Queue,
    local_crs,
    parallel: bool,
    max_workers: int | None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:

    splitted: list[gpd.GeoDataFrame] = []
    roads_all: list[gpd.GeoDataFrame] = []

    if not parallel:
        while True:
            try:
                func, task, kwargs = task_queue.get_nowait()
            except queue.Empty:
                break

            result: dict = func(task, **kwargs)

            for nt in result.get("new_tasks", []) or []:
                task_queue.put(nt)

            gen = result.get("generation")
            if isinstance(gen, gpd.GeoDataFrame) and len(gen) > 0:
                splitted.append(gen)

            roads = result.get("generated_roads")
            if isinstance(roads, gpd.GeoDataFrame) and len(roads) > 0:
                roads_all.append(roads)

        return _merge_gdfs(splitted, local_crs), _merge_gdfs(roads_all, local_crs)

    workers = int(max_workers or multiprocessing.cpu_count())

    future_to_nothing: dict[concurrent.futures.Future, None] = {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        while True:
            while len(future_to_nothing) < workers:
                try:
                    func, task, kwargs = task_queue.get_nowait()
                except queue.Empty:
                    break
                future = executor.submit(func, task, **kwargs)
                future_to_nothing[future] = None

            if not future_to_nothing:
                break

            done, _ = concurrent.futures.wait(
                future_to_nothing.keys(),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )

            for fut in done:
                future_to_nothing.pop(fut, None)
                result: dict = fut.result()

                for nt in result.get("new_tasks", []) or []:
                    task_queue.put(nt)

                gen = result.get("generation")
                if isinstance(gen, gpd.GeoDataFrame) and len(gen) > 0:
                    splitted.append(gen)

                roads = result.get("generated_roads")
                if isinstance(roads, gpd.GeoDataFrame) and len(roads) > 0:
                    roads_all.append(roads)

    return _merge_gdfs(splitted, local_crs), _merge_gdfs(roads_all, local_crs)
