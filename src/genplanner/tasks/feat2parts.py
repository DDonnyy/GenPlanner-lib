import pandas as pd
from shapely import LineString, Point, Polygon

from genplanner.tasks.polygon_splitter import split_polygon
from genplanner.utils import polygon_angle, rotate_coords


def gdf_splitter(task, **kwargs):
    gdf, areas_dict, roads_width, fixed_zones = task
    n_areas = len(areas_dict)
    generated_zones = []
    generated_roads = []
    local_crs = gdf.crs
    for ind, row in gdf.iterrows():
        polygon = row.geometry
        pivot_point = polygon.centroid
        angle_rad_to_rotate = polygon_angle(polygon)
        if len(fixed_zones) > 0:
            fixed_zones_in_poly = fixed_zones[fixed_zones.within(polygon)]
            if len(fixed_zones_in_poly) > 0:
                fixed_zones_in_poly["geometry"] = fixed_zones_in_poly["geometry"].apply(
                    lambda x: Point(rotate_coords(x.coords, pivot_point, -angle_rad_to_rotate))
                )
                fixed_zones_in_poly = fixed_zones_in_poly.groupby("fixed_zone").agg({"geometry": list})
                fixed_zones_in_poly = fixed_zones_in_poly["geometry"].to_dict()
            else:
                fixed_zones_in_poly = None
        else:
            fixed_zones_in_poly = None
        polygon = Polygon(rotate_coords(polygon.exterior.coords, pivot_point, -angle_rad_to_rotate))
        zones, roads = split_polygon(
            polygon_to_split=polygon,
            zone_ratios=areas_dict,
            point_radius=poisson_n_radius.get(n_areas, 0.1),
            local_crs=local_crs,
            zone_fixed_point=fixed_zones_in_poly,
            dev=kwargs.get("dev_mod"),
        )

        if not zones.empty:
            zones.geometry = zones.geometry.apply(
                lambda x: Polygon(rotate_coords(x.exterior.coords, pivot_point, angle_rad_to_rotate))
            )
        if not roads.empty:
            roads.geometry = roads.geometry.apply(
                lambda x: LineString(rotate_coords(x.coords, pivot_point, angle_rad_to_rotate))
            )
        generated_zones.append(zones)
        generated_roads.append(roads)

    roads = pd.concat(generated_roads, ignore_index=True)
    zones = pd.concat(generated_zones, ignore_index=True)

    roads["road_lvl"] = "undefined"  # TODO kwargs??
    roads["roads_width"] = roads_width if roads_width is not None else roads_width_def.get("local road")
    return {"generation": zones, "generated_roads": roads}
