import concurrent.futures
import multiprocessing
import os
import time
from datetime import datetime
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

    def __init__(
        self,
        features_gdf: gpd.GeoDataFrame,
        roads_gdf: gpd.GeoDataFrame | None = None,
        roads_extend_distance: float = 5,
        exclude_gdf: gpd.GeoDataFrame | None = None,
        exclude_buffer: float = 5,
        existing_terr_zones: gpd.GeoDataFrame | None = None,
        existing_tz_fill_ratio: float = 0.8,
        existing_tz_merge_radius=50,
        simplify_geometry_value=0.01,
        parallel=True,
        parallel_max_workers=None,
        rust_write_logs=False,
        run_name=None,
    ):
        """
        - simplify_geometry_value: do not affect output zones geometry, only used for internal calculations.
         Higher value can speed up the process, but can lead to less accurate results. Value is in normalized 0-1 space.
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
            self.run_name = now.strftime("gp%d%m%y_%H:%M")
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
    ) -> Polygon | MultiPolygon:

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
        kwargs.update(
            {
                "rust_write_logs": self.rust_write_logs,
                "simplify": self.simplify_geometry_value,
                "local_crs": self.local_crs.to_epsg(),
                "run_name": self.run_name,
            }
        )

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

    RelationMatrixArg = ZoneRelationMatrix | Literal["empty", "default"] | None

    def features2terr_zones(
        self,
        funczone: FunctionalZone = basic_func_zone,
        relation_matrix: RelationMatrixArg = "default",
        terr_zones_fix_points: gpd.GeoDataFrame = None,
    ) -> (gpd.GeoDataFrame, gpd.GeoDataFrame):
        return self._features2terr_zones(funczone, relation_matrix, terr_zones_fix_points, split_further=False)

    def features2terr_zones2blocks(
        self,
        funczone: FunctionalZone = basic_func_zone,
        relation_matrix: RelationMatrixArg = "default",
        terr_zones_fix_points: gpd.GeoDataFrame = None,
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:

        return self._features2terr_zones(funczone, relation_matrix, terr_zones_fix_points, split_further=True)

    def _features2terr_zones(
        self,
        funczone: FunctionalZone,
        relation_matrix: RelationMatrixArg,
        terr_zones_fix_points: gpd.GeoDataFrame,
        split_further: bool,
    ) -> (gpd.GeoDataFrame, gpd.GeoDataFrame):

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


def split_queue(
    task_queue: multiprocessing.Queue, local_crs, parallel, max_workers
) -> (gpd.GeoDataFrame, gpd.GeoDataFrame):
    splitted = []
    roads_all = []

    if not parallel:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    else:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers)

    with executor:
        future_to_task = {}
        while True:
            while not task_queue.empty() and len(future_to_task) < executor._max_workers:
                func, task, kwargs = task_queue.get_nowait()
                future = executor.submit(func, task, **kwargs)

                future_to_task[future] = task

            done, _ = concurrent.futures.wait(future_to_task.keys(), return_when=concurrent.futures.FIRST_COMPLETED)

            for future in done:
                future_to_task.pop(future)
                result: dict = future.result()
                new_tasks = result.get("new_tasks", [])
                if len(new_tasks) > 0:
                    for func, new_task, kwargs in new_tasks:
                        task_queue.put((func, new_task, kwargs))

                generated_zones = result.get("generation", gpd.GeoDataFrame())

                if len(generated_zones) > 0:
                    splitted.append(generated_zones)

                generated_roads = result.get("generated_roads", gpd.GeoDataFrame())
                if len(generated_roads) > 0:
                    roads_all.append(generated_roads)

            time.sleep(0.01)
            if not future_to_task and task_queue.empty():
                break

    if len(roads_all) > 0:
        roads_to_return = gpd.GeoDataFrame(pd.concat(roads_all, ignore_index=True), crs=local_crs, geometry="geometry")
    else:
        roads_to_return = gpd.GeoDataFrame()
    return (
        gpd.GeoDataFrame(pd.concat(splitted, ignore_index=True), crs=local_crs, geometry="geometry"),
        roads_to_return,
    )
