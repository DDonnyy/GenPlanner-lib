import geopandas as gpd
import numpy as np

from genplanner import config
from genplanner.errors import GenPlannerArgumentError
from genplanner.tasks.polygon_splitter import split_polygon
from genplanner.zones import BasicZone

roads_width_def = config.roads_width_def.copy()


def multi_feature2blocks_initial(task, **kwargs):
    (poly_gdf,) = task

    if not isinstance(poly_gdf, gpd.GeoDataFrame):
        raise GenPlannerArgumentError(f"poly_gdf wrong dtype {type(poly_gdf)}", run_name=f"{kwargs['run_name']}_f2b")
    if "territory_zone" not in poly_gdf.columns:
        raise GenPlannerArgumentError(
            f"`territory_zone` column not presented in provided gdf", run_name=f"{kwargs['run_name']}_f2b"
        )

    new_tasks = []

    for ind, row in poly_gdf.iterrows():
        zone_kwargs = kwargs.copy()
        zone_kwargs.update({"territory_zone": row.territory_zone, "run_name": f"{kwargs['run_name']}_f2b{ind}"})

        geometry = row.geometry
        target_area = geometry.area
        min_block_area = row.territory_zone.min_block_area

        max_delimiter = 6
        temp_area = min_block_area
        delimiters = []

        # Ищем максимальную площадь больше чем таргет
        while temp_area < target_area:
            temp_area *= max_delimiter
            delimiters.append(max_delimiter)

        if len(delimiters) == 0:
            new_tasks.append(
                (
                    feature2blocks_splitter,
                    (geometry, [1], min_block_area, 1, [roads_width_def.get("local road")]),
                    zone_kwargs,
                )
            )
            continue

        min_split = 1 if len(delimiters) == 1 else 2
        i = 0
        # Убираем деления, пока не приблизимся к площади
        while temp_area > target_area:
            if delimiters[i] > min_split:
                delimiters[i] -= 1
            else:
                i += 1
            temp_area = min_block_area * np.prod(delimiters)
        # Возвращаем последнее удаление
        delimiters[i] += 1

        min_width = int(roads_width_def.get("regulated highway") * 0.66)
        max_width = roads_width_def.get("local road")
        roads_widths = np.linspace(min_width, max_width, len(delimiters))

        # Добавление задачи
        new_tasks.append(
            (feature2blocks_splitter, (geometry, delimiters, min_block_area, 1, roads_widths), zone_kwargs)
        )
    return {"new_tasks": new_tasks}


def feature2blocks_splitter(task, **kwargs):
    polygon, delimeters, min_area, deep, roads_widths = task

    local_crs = kwargs["local_crs"]

    if deep < 1:
        raise ValueError(f"deep must be >= 1, got {deep}")
    if deep - 1 >= len(roads_widths):
        raise ValueError(f"roads_widths too short: need index {deep - 1}, len={len(roads_widths)}")
    if deep != len(delimeters) and deep - 1 >= len(delimeters):
        raise ValueError(f"delimeters too short: need index {deep - 1}, len={len(delimeters)}")

    if deep == len(delimeters):
        n_areas = min(8, int(polygon.area // min_area))
    else:
        n_areas = delimeters[deep - 1]
        n_areas = min(n_areas, int(polygon.area // min_area))

    if n_areas in [0, 1]:
        data = {key: [value] for key, value in kwargs.items() if key in ["territory_zone", "func_zone", "gen_plan"]}
        blocks = gpd.GeoDataFrame(data=data, geometry=[polygon], crs=local_crs)
        return {"generation": blocks}

    areas_dict = {BasicZone(name=f"block_{i}"): 1.0 / n_areas for i in range(n_areas)}

    blocks, roads = split_polygon(
        polygon_to_split=polygon,
        zone_ratios=areas_dict,
        zone_neighbors=[],
        zone_forbidden=[],
        zone_fixed_point={},
        local_crs=local_crs,
        run_name=f"{kwargs['run_name']}_d{deep}n{n_areas}",
        geom_simplify_tol=kwargs["simplify"],
        allow_multipolygon=False,
        write_logs=kwargs["rust_write_logs"],
        seed=kwargs.get("seed", None),
        sites_multiplier=kwargs.get("sites_multiplier", 5),
    )
    road_lvl = "local road"
    roads["road_lvl"] = f"{road_lvl}, level {deep}"
    roads["roads_width"] = roads_widths[deep - 1]
    if deep == len(delimeters):
        data = {
            key: [value] * len(blocks)
            for key, value in kwargs.items()
            if key in ["territory_zone", "func_zone", "gen_plan"]
        }
        blocks = gpd.GeoDataFrame(data=data, geometry=blocks.geometry, crs=local_crs)
        return {"generation": blocks, "generated_roads": roads}
    else:
        deep = deep + 1
        blocks = blocks.geometry
        tasks = []
        for poly in blocks:
            if poly is not None:
                tasks.append((feature2blocks_splitter, (poly, delimeters, min_area, deep, roads_widths), kwargs))

        return {"new_tasks": tasks, "generated_roads": roads}
