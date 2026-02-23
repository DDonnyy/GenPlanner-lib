import geopandas as gpd
import pandas as pd
import pulp
from shapely import LineString, Point, Polygon
from shapely.ops import nearest_points

from genplanner import config
from genplanner.errors import GenplannerInfeasibleMultiFeatureError
from genplanner.tasks.feat2blocks import multi_feature2blocks_initial
from genplanner.tasks.polygon_splitter import split_polygon
from genplanner.utils import (
    elastic_wrap,
    polygon_angle,
    rotate_coords,
)
from genplanner.zone_relations.relation_matrix import ZoneRelationMatrix
from genplanner.zones import FunctionalZone

logger = config.logger

roads_width_def = config.roads_width_def.copy()


def filter_terr_zone(terr_zones: pd.DataFrame, area) -> pd.DataFrame:
    def recalculate_ratio(data, area):
        data["ratio"] = data["ratio"] / (data["ratio"].sum())
        data["required_area"] = area * data["ratio"]
        data["good"] = (data["min_block_area"] * 0.8) < data["required_area"]
        return data

    terr_zones = recalculate_ratio(terr_zones, area)
    while not terr_zones["good"].all():
        terr_zones = terr_zones[terr_zones["good"]].copy()
        logger.debug(f"removed terr_zones {terr_zones[~terr_zones['good']]}")
        terr_zones = recalculate_ratio(terr_zones, area)
    return terr_zones


def multi_feature2terr_zones_initial(task, **kwargs):

    initial_gdf, func_zone, relation_matrix, fixed_terr_zones, split_further = task
    initial_gdf: gpd.GeoDataFrame
    func_zone: FunctionalZone
    relation_matrix: ZoneRelationMatrix
    fixed_terr_zones: gpd.GeoDataFrame
    split_further: bool
    run_name = f"{kwargs['run_name']}_mf2tz"
    local_crs = kwargs["local_crs"]
    initial_gdf["feature_area"] = initial_gdf.area

    territory_union = elastic_wrap(initial_gdf)

    terr_zones = pd.DataFrame.from_dict(
        {terr_zone: [ratio, terr_zone.min_block_area] for terr_zone, ratio in func_zone.zones_ratio.items()},
        orient="index",
        columns=["ratio", "min_block_area"],
    )
    terr_zones = filter_terr_zone(terr_zones, initial_gdf["feature_area"].sum())
    terr_zone_list = tuple(terr_zones.index)
    relation_matrix = relation_matrix.subset(terr_zone_list, strict=False)

    if fixed_terr_zones is None:
        fixed_terr_zones = gpd.GeoDataFrame()

    if len(fixed_terr_zones) > 0:
        fixed_zones_in_poly = fixed_terr_zones[fixed_terr_zones.within(territory_union)].copy()
        if len(fixed_zones_in_poly) > 0:
            fixed_zones_in_poly = fixed_zones_in_poly.set_index("fixed_zone")["geometry"].to_dict()
            fixed_zones_in_poly = {z: geom for z, geom in fixed_zones_in_poly.items() if z in terr_zone_list}
        else:
            fixed_zones_in_poly = {}
    else:
        fixed_zones_in_poly = {}

    proxy_zones, _ = split_polygon(
        polygon_to_split=territory_union,
        zone_ratios=terr_zones["ratio"].to_dict(),
        zone_neighbors=relation_matrix.zone_neighbors(),
        zone_forbidden=relation_matrix.zone_forbidden(),
        zone_fixed_point=fixed_zones_in_poly,
        local_crs=local_crs,
        run_name=f"proxy{run_name}",
        geom_simplify_tol=kwargs["simplify"],
        allow_multipolygon=True,
        write_logs=kwargs["rust_write_logs"],
        seed=kwargs.get("seed", None),
        sites_multiplier=kwargs.get("sites_multiplier", 5),
    )

    upd_fix_terr_zones = proxy_zones.copy()
    upd_fix_terr_zones.geometry = upd_fix_terr_zones.geometry.centroid
    if len(fixed_terr_zones) > 0:
        upd_fix_terr_zones = upd_fix_terr_zones.merge(
            fixed_terr_zones, left_on="zone", right_on="fixed_zone", how="left", suffixes=("", "_fixed")
        )
        upd_fix_terr_zones["geometry"] = upd_fix_terr_zones["geometry_fixed"].combine_first(
            upd_fix_terr_zones["geometry"]
        )
        upd_fix_terr_zones = upd_fix_terr_zones.drop(columns=["geometry_fixed", "fixed_zone"])

    division = initial_gdf.sjoin(proxy_zones)

    terr_zones = terr_zones.reset_index(names="zone")
    terr_zones["required_area"] = terr_zones["required_area"] * 0.999

    zone_capacity = division.groupby(level=0)["feature_area"].first().to_dict()
    zone_permitted = set(division["zone"].items())

    target_areas = terr_zones.set_index("zone")["required_area"].to_dict()

    model = pulp.LpProblem("Territorial_Zoning", pulp.LpMinimize)

    x = {(i, z): pulp.LpVariable(f"feature index {i} zone type {z}", lowBound=0) for (i, z) in zone_permitted}
    y = {(i, z): pulp.LpVariable(f"y_{i}_{z}", cat="Binary") for (i, z) in zone_permitted}

    for i in division.index.unique():
        model += (
            pulp.lpSum(x[i, z] for z in terr_zones["zone"] if (i, z) in x) <= zone_capacity[i],
            f"Capacity_feature_{i}",
        )
    for i, z in x:

        model += x[i, z] <= zone_capacity[i] * y[i, z], f"MaxIfAssigned_{i}_{z}"

    for z in terr_zones["zone"]:
        model += (
            pulp.lpSum(x[i, z] for i in division.index.unique() if (i, z) in x) >= target_areas[z],
            f"TargetArea_{z}",
        )

    if len(fixed_terr_zones) > 0:
        fixed_terr_zones["zone_name"] = fixed_terr_zones["fixed_zone"].apply(lambda x: x.name)
        zone_strongly_fixed = set(initial_gdf.sjoin(fixed_terr_zones)[["zone_name"]].itertuples(name=None))
        for i, z in zone_strongly_fixed:
            if (i, z) in x:
                model += x[i, z] >= 1e-3, f"StronglyFixed_{i}_{z}"

    model.solve(pulp.PULP_CBC_CMD(msg=True, timeLimit=20, gapRel=0.01))

    if model.status == pulp.LpStatusInfeasible:
        if kwargs.get("infeasible", 0) == 5:
            raise GenplannerInfeasibleMultiFeatureError("Cannot solve task in 5 tries.")
        kwargs.update({"run_name": f"{run_name}_infeasible", "infeasible": kwargs.get("infeasible", 0) + 1})
        return {"new_tasks": [(multi_feature2terr_zones_initial, task, kwargs)]}

    allocations = []
    for (i, z), var in x.items():
        val = var.varValue
        if val and val > 0:
            allocations.append((i, z, round(val, 2)))
    del x, y
    allocations = pd.DataFrame(allocations, columns=["zone_index", "territorial_zone", "assigned_area"])

    kwargs.update({"func_zone": func_zone, "run_name": run_name})

    ready_for_blocks: list[gpd.GeoDataFrame] = []
    new_tasks: list[tuple] = []

    alloc_pos = allocations.loc[allocations["assigned_area"] > 0].copy()
    groups = dict(tuple(alloc_pos.groupby("zone_index", sort=False)))
    initial_sel = initial_gdf.loc[list(groups.keys())]

    for ind, zone_row in initial_sel.iterrows():
        zone_polygon = zone_row.geometry
        terr_zones_in_poly = groups[ind]

        n = len(terr_zones_in_poly)
        if n == 1:
            terr_zone = terr_zones_in_poly.iloc[0]["territorial_zone"]
            ready_for_blocks.append(
                gpd.GeoDataFrame(
                    {"territory_zone": [terr_zone]},
                    geometry=[zone_polygon],
                    crs=local_crs,
                )
            )
            continue

        if n > 1:
            zone_area_total = zone_row["feature_area"]
            zones_ratio_dict = {
                row["territorial_zone"]: row["assigned_area"] / zone_area_total
                for _, row in terr_zones_in_poly.iterrows()
            }

            task_gdf = gpd.GeoDataFrame(geometry=[zone_polygon], crs=local_crs)
            task_func_zone = FunctionalZone(zones_ratio_dict, name=func_zone.name)

            needed_zones = list(zones_ratio_dict.keys())
            task_fixed_terr_zones = upd_fix_terr_zones.loc[upd_fix_terr_zones["zone"].isin(needed_zones)].copy()

            task_fixed_terr_zones = task_fixed_terr_zones.rename(columns={"zone": "fixed_zone"})
            task_fixed_terr_zones["geometry"] = task_fixed_terr_zones["geometry"].apply(
                lambda fix_p: nearest_points(fix_p, zone_polygon.buffer(-0.1, resolution=1))[1]
            )

            task_fixed_terr_zones = task_fixed_terr_zones.loc[
                ~task_fixed_terr_zones["geometry"].duplicated(keep="first")
            ]
            task_relation_matrix = relation_matrix.subset(needed_zones)

            task_kwargs = kwargs.copy()
            task_kwargs.update({"run_name": f"{run_name}_{ind}"})

            new_tasks.append(
                (
                    feature2terr_zones_initial,
                    (task_gdf, task_func_zone, task_relation_matrix, task_fixed_terr_zones, split_further),
                    task_kwargs,
                )
            )

    if len(ready_for_blocks) > 0:
        block_splitter_gdf = pd.concat(ready_for_blocks)
    else:
        block_splitter_gdf = gpd.GeoDataFrame()

    if split_further:
        if len(block_splitter_gdf) > 0:
            new_tasks.append((multi_feature2blocks_initial, (block_splitter_gdf,), kwargs))
        return {"new_tasks": new_tasks}
    else:
        if len(block_splitter_gdf) > 0:
            block_splitter_gdf["func_zone"] = func_zone
        return {"new_tasks": new_tasks, "generation": block_splitter_gdf}


def feature2terr_zones_initial(task, **kwargs):
    initial_gdf, func_zone, relation_matrix, fixed_terr_zones, split_further = task
    initial_gdf: gpd.GeoDataFrame
    func_zone: FunctionalZone
    relation_matrix: ZoneRelationMatrix
    fixed_terr_zones: gpd.GeoDataFrame
    split_further: bool
    run_name = f"{kwargs['run_name']}_f2tz"
    local_crs = kwargs["local_crs"]

    polygon = initial_gdf.iloc[0].geometry
    area = polygon.area

    terr_zones = pd.DataFrame.from_dict(
        {terr_zone: [ratio, terr_zone.min_block_area] for terr_zone, ratio in func_zone.zones_ratio.items()},
        orient="index",
        columns=["ratio", "min_block_area"],
    )

    terr_zones = filter_terr_zone(terr_zones, area)
    terr_zone_list = tuple(terr_zones.index)
    relation_matrix = relation_matrix.subset(terr_zone_list, strict=False)

    if len(terr_zones) == 0:
        profile_terr = max(func_zone.zones_ratio.items(), key=lambda x: x[1])[0]
        data = {"territory_zone": [profile_terr], "func_zone": [func_zone], "geometry": [polygon]}
        return {"generation": gpd.GeoDataFrame(data=data, geometry="geometry", crs=local_crs)}

    if fixed_terr_zones is None:
        fixed_terr_zones = gpd.GeoDataFrame()

    if len(fixed_terr_zones) > 0:
        fixed_zones_in_poly = fixed_terr_zones[fixed_terr_zones.intersects(polygon)].copy()
        if len(fixed_zones_in_poly) > 0:
            fixed_zones_in_poly = fixed_zones_in_poly.groupby("fixed_zone", as_index=False).agg({"geometry": "first"})
            fixed_zones_in_poly = fixed_zones_in_poly.set_index("fixed_zone")["geometry"].to_dict()
            fixed_zones_in_poly = {z: geom for z, geom in fixed_zones_in_poly.items() if z in terr_zone_list}
        else:
            fixed_zones_in_poly = None
    else:
        fixed_zones_in_poly = None

    if len(terr_zones) > 1:
        zones, roads = split_polygon(
            polygon_to_split=polygon,
            zone_ratios=terr_zones["ratio"].to_dict(),
            zone_neighbors=relation_matrix.zone_neighbors(),
            zone_forbidden=relation_matrix.zone_forbidden(),
            zone_fixed_point=fixed_zones_in_poly,
            local_crs=local_crs,
            run_name=run_name,
            geom_simplify_tol=kwargs["simplify"],
            allow_multipolygon=False,
            write_logs=kwargs["rust_write_logs"],
            seed=kwargs.get("seed", None),
            sites_multiplier=kwargs.get("sites_multiplier", 5),
        )
    else:
        data = {"zone": [terr_zones.index[0]], "func_zone": [func_zone], "geometry": [polygon]}
        zones = gpd.GeoDataFrame(data=data, geometry="geometry", crs=local_crs)
        roads = gpd.GeoDataFrame()

    road_lvl = "regulated highway"
    roads["road_lvl"] = road_lvl
    roads["roads_width"] = roads_width_def.get("regulated highway")

    if not split_further:
        zones["func_zone"] = func_zone
        if len(zones) > 0:
            zones["territory_zone"] = zones["zone"]
            zones = zones[["func_zone", "territory_zone", "geometry"]]
        return {"generation": zones, "generated_roads": roads}

    # if split further
    kwargs.update({"func_zone": func_zone})
    kwargs.update({"from": "feature2terr_zones_initial"})
    if len(zones) > 0:
        zones["territory_zone"] = zones["zone"]
    task = [(multi_feature2blocks_initial, (zones,), kwargs)]
    return {"new_tasks": task, "generated_roads": roads}
