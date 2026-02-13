import math

import geopandas as gpd
import numpy as np
import pandas as pd
from genplanner._rust import optimize_territory_zoning
from pyproj import CRS
from shapely import Point
from shapely.geometry import LineString, MultiPolygon, Polygon

from genplanner._config import config
from genplanner.utils import (
    denormalize_coords,
    generate_points,
    normalize_coords,
    polygon_angle,
    rotate_coords,
)
from genplanner.zoning.abc_zone import BaseZone

roads_width_def = config.roads_width_def.copy()

GLOBAL_POINT_POOL_SEED = config.point_pool_seed
GLOBAL_POINT_POOL_SIZE = config.point_pool_size
_rng_global = np.random.default_rng(GLOBAL_POINT_POOL_SEED)

GLOBAL_POINT_POOL_U01 = _rng_global.random((GLOBAL_POINT_POOL_SIZE, 2), dtype=np.float64)


class SplitPolygonValidationError(ValueError):
    """Raised when _split_polygon input arguments fail fast validation."""

    def __init__(self, field: str, message: str):
        super().__init__(f"{field}: {message}")
        self.field = field
        self.message = message


class MultiPolygonSplitError(Exception):
    """Raised when optimizer returns MultiPolygon and allow_multipolygon is False."""

    pass


def _is_finite_number(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _validate_polygon_no_holes(polygon_to_split):
    if not isinstance(polygon_to_split, Polygon):
        raise SplitPolygonValidationError(
            "polygon_to_split",
            f"expected shapely.geometry.Polygon, got {type(polygon_to_split).__name__}",
        )
    if len(polygon_to_split.interiors) != 0:
        raise SplitPolygonValidationError("polygon_to_split", "polygon must NOT have holes")
    if polygon_to_split.is_empty:
        raise SplitPolygonValidationError("polygon_to_split", "polygon is empty")
    if not polygon_to_split.is_valid:
        raise SplitPolygonValidationError("polygon_to_split", "polygon is not valid (self-intersection or similar)")


def _validate_zone_ratios(zone_ratios):
    if not isinstance(zone_ratios, dict):
        raise SplitPolygonValidationError(
            "zone_ratios", f"expected dict[BaseZone, float], got {type(zone_ratios).__name__}"
        )
    if not zone_ratios:
        raise SplitPolygonValidationError("zone_ratios", "must be non-empty")

    for k, v in zone_ratios.items():
        if not isinstance(k, BaseZone):
            raise SplitPolygonValidationError(
                "zone_ratios", f"all keys must be BaseZone, got key {k!r} of type {type(k).__name__}"
            )
        if not _is_finite_number(v):
            raise SplitPolygonValidationError(
                "zone_ratios", f"all values must be finite numbers (float), got {v!r} of type {type(v).__name__}"
            )
        if v <= 0.0:
            raise SplitPolygonValidationError("zone_ratios", f"ratio for {k!r} must be > 0, got {v}")


def _validate_zone_pairs(field: str, pairs):
    if not isinstance(pairs, list):
        raise SplitPolygonValidationError(
            field,
            f"expected list[tuple[BaseZone, BaseZone]], got {type(pairs).__name__}",
        )
    if not pairs:
        return

    seen_undirected: set[frozenset["BaseZone"]] = set()

    for i, item in enumerate(pairs):
        if not isinstance(item, tuple) or len(item) != 2:
            raise SplitPolygonValidationError(
                field,
                f"item #{i} must be tuple(BaseZone, BaseZone), got {item!r}",
            )
        a, b = item
        if not isinstance(a, BaseZone) or not isinstance(b, BaseZone):
            raise SplitPolygonValidationError(
                field,
                f"item #{i} must contain BaseZone objects, got " f"({type(a).__name__}, {type(b).__name__})",
            )
        if a == b:
            raise SplitPolygonValidationError(
                field,
                f"item #{i}: self-pair ({a!r}, {b!r}) is not allowed",
            )
        key = frozenset((a, b))
        if key in seen_undirected:
            raise SplitPolygonValidationError(
                field,
                f"duplicate undirected pair found at item #{i}: ({a!r}, {b!r}) "
                f"(duplicates also include reversed order)",
            )
        seen_undirected.add(key)


def _validate_zone_fixed_point(zone_fixed_point):
    if not isinstance(zone_fixed_point, dict):
        raise SplitPolygonValidationError(
            "zone_fixed_point",
            f"expected dict[BaseZone, Point], got {type(zone_fixed_point).__name__}",
        )

    if not zone_fixed_point:
        return {}

    for k, v in zone_fixed_point.items():
        if not isinstance(k, BaseZone):
            raise SplitPolygonValidationError(
                "zone_fixed_point",
                f"key must be BaseZone, got {type(k).__name__}",
            )

        if not isinstance(v, Point):
            raise SplitPolygonValidationError(
                "zone_fixed_point",
                f"value for {k!r} must be shapely.geometry.Point, got {type(v).__name__}",
            )

        if v.is_empty or not v.is_valid:
            raise SplitPolygonValidationError(
                "zone_fixed_point",
                f"Point for {k!r} must be non-empty and valid",
            )


def _validate_local_crs(local_crs):
    if not isinstance(local_crs, CRS):
        raise SplitPolygonValidationError("local_crs", f"expected pyproj.CRS, got {type(local_crs).__name__}")


def _validate_bool(field: str, v):
    if not isinstance(v, bool):
        raise SplitPolygonValidationError(field, f"expected bool, got {type(v).__name__}")


def _validate_run_name(run_name):
    if not isinstance(run_name, str):
        raise SplitPolygonValidationError("run_name", f"expected str, got {type(run_name).__name__}")
    if not run_name.strip():
        raise SplitPolygonValidationError("run_name", "must be a non-empty string")


def _validate_split_polygon_args(
    polygon_to_split: Polygon,
    zone_ratios: dict[BaseZone, float],
    zone_neighbors: list[tuple[BaseZone, BaseZone]],
    zone_forbidden: list[tuple[BaseZone, BaseZone]],
    zone_fixed_point: dict[BaseZone, Point],
    local_crs: CRS,
    run_name: str,
    normalize_rotation: bool = True,
    allow_multipolygon=False,
    write_logs=False,
) -> None:
    _validate_polygon_no_holes(polygon_to_split)

    _validate_zone_ratios(zone_ratios)
    _validate_zone_pairs("zone_neighbors", zone_neighbors)
    _validate_zone_pairs("zone_forbidden", zone_forbidden)
    _validate_zone_fixed_point(zone_fixed_point)
    _validate_local_crs(local_crs)
    _validate_run_name(run_name)
    _validate_bool("allow_multipolygon", allow_multipolygon)
    _validate_bool("write_logs", write_logs)
    _validate_bool("normalize_rotation", normalize_rotation)

    declared = set(zone_ratios.keys())
    for a, b in zone_neighbors:
        if a not in declared or b not in declared:
            raise SplitPolygonValidationError(
                "zone_neighbors",
                f"zone {a!r} or {b!r} not present in zone_ratios keys",
            )
    for a, b in zone_forbidden:
        if a not in declared or b not in declared:
            raise SplitPolygonValidationError(
                "zone_forbidden",
                f"zone {a!r} or {b!r} not present in zone_ratios keys",
            )
    for z in zone_fixed_point.keys():
        if z not in declared:
            raise SplitPolygonValidationError(
                "zone_fixed_point",
                f"zone {z!r} not present in zone_ratios keys",
            )


def _create_polygons(site2idx, point2zone, idx2vtxv, vtxv2xy):
    poly_coords = []
    poly_sites = []
    for i_site in range(len(site2idx) - 1):
        if point2zone[i_site] == np.iinfo(np.uint32).max:
            continue

        num_vtx_in_site = site2idx[i_site + 1] - site2idx[i_site]
        if num_vtx_in_site == 0:
            continue

        vtx2xy = []
        for i_vtx in range(num_vtx_in_site):  # collecting poly
            i_vtxv = idx2vtxv[site2idx[i_site] + i_vtx]  # founding vertex id
            vtx2xy.append((vtxv2xy[i_vtxv * 2], vtxv2xy[i_vtxv * 2 + 1]))  # adding vertex xy to poly
        poly_sites.append(point2zone[i_site])
        poly_coords.append(Polygon(vtx2xy))

    return poly_coords, poly_sites


def _allocate_sites_sqrt_ratio(
    zone_ratios: dict["BaseZone", float],
    total_sites: int,
    min_per_zone: int = 2,
) -> dict["BaseZone", int]:
    """
    Allocate total_sites across zones proportional to sqrt(ratio),
    with each zone having at least min_per_zone.
    Returns dict {zone: count}.
    """
    zones = list(zone_ratios.keys())
    Z = len(zones)

    if Z == 0:
        raise SplitPolygonValidationError("zone_ratios", "must be non-empty")
    if total_sites < min_per_zone * Z:
        raise SplitPolygonValidationError(
            "sites_allocation",
            f"total_sites={total_sites} is too small for min_per_zone={min_per_zone} and Z={Z} "
            f"(need at least {min_per_zone * Z})",
        )

    ratios = np.array([float(zone_ratios[z]) for z in zones], dtype=np.float64)
    w = np.sqrt(ratios)
    w = w / w.sum()

    counts = np.full(Z, min_per_zone, dtype=np.int64)
    remaining = total_sites - min_per_zone * Z
    if remaining == 0:
        return {z: int(c) for z, c in zip(zones, counts)}

    target = w * remaining
    add = np.floor(target).astype(np.int64)
    counts += add
    leftover = remaining - int(add.sum())

    if leftover > 0:
        remainders = target - add
        order = np.argsort(-remainders)
        for k in range(leftover):
            counts[order[k]] += 1

    return {z: int(c) for z, c in zip(zones, counts)}


def _sample_points_from_global_pool(
    total_sites: int,
    seed: int,
) -> np.ndarray:
    """
    Deterministically choose total_sites points from GLOBAL_POINT_POOL_U01,
    allowing repeats if total_sites > pool size.
    Returns array shape (total_sites, 2).
    """
    pool = GLOBAL_POINT_POOL_U01
    pool_n = pool.shape[0]

    rng = np.random.default_rng(seed)

    if total_sites <= pool_n:
        idx = rng.permutation(pool_n)[:total_sites]
        return pool[idx].copy()

    idx = rng.integers(0, pool_n, size=total_sites, endpoint=False)
    return pool[idx].copy()


def split_polygon(
    polygon_to_split: Polygon,
    zone_ratios: dict[BaseZone, float],
    zone_neighbors: list[tuple[BaseZone, BaseZone]],
    zone_forbidden: list[tuple[BaseZone, BaseZone]],
    zone_fixed_point: dict[BaseZone, Point],
    local_crs: CRS,
    run_name: str,
    normalize_rotation: bool = True,
    allow_multipolygon=False,
    write_logs=False,
) -> (gpd.GeoDataFrame, gpd.GeoDataFrame):

    _validate_split_polygon_args(
        polygon_to_split=polygon_to_split,
        zone_ratios=zone_ratios,
        zone_neighbors=zone_neighbors,
        zone_forbidden=zone_forbidden,
        zone_fixed_point=zone_fixed_point,
        local_crs=local_crs,
        run_name=run_name,
        allow_multipolygon=allow_multipolygon,
        write_logs=write_logs,
        normalize_rotation=normalize_rotation,
    )

    areas_init = pd.DataFrame(list(zone_ratios.items()), columns=["zone", "ratio"])
    fallback_zone_name = areas_init.loc[areas_init["ratio"].idxmax(), "zone"]

    zone_to_idx = {zone_name: idx for idx, zone_name in enumerate(areas_init["zone"])}

    pivot_point = None
    angle_rad_to_rotate = None
    if normalize_rotation:
        pivot_point = polygon_to_split.centroid
        angle_rad_to_rotate = polygon_angle(polygon_to_split)
        polygon_to_split = Polygon(rotate_coords(polygon_to_split.exterior.coords, pivot_point, -angle_rad_to_rotate))

    bounds = polygon_to_split.bounds

    normalized_polygon = Polygon(normalize_coords(polygon_to_split.exterior.coords, bounds))

    fixed_points = []
    for zone, point in zone_fixed_point.items():
        room_idx = zone_to_idx[zone]
        if normalize_rotation:
            point = Point(rotate_coords(point.coords, pivot_point, -angle_rad_to_rotate))
        xy = normalize_coords(point.coords, bounds)
        fixed_points.append((xy[0][0], xy[0][1], room_idx))

    full_area = normalized_polygon.area
    areas = areas_init.copy()
    areas["ratio"] = areas["ratio"] / (areas["ratio"].sum())
    areas["area"] = areas["ratio"] * full_area

    Z = len(zone_ratios)
    total_sites = 5 * Z

    zone2count = _allocate_sites_sqrt_ratio(zone_ratios=zone_ratios, total_sites=total_sites, min_per_zone=2)
    counts_by_idx = np.zeros(Z, dtype=np.int64)
    for z, cnt in zone2count.items():
        counts_by_idx[zone_to_idx[z]] = cnt
    base_point2zone = np.repeat(np.arange(Z, dtype=np.int64), counts_by_idx)
    run_seed = hash(run_name) & 0xFFFFFFFF

    attempts = 10
    best_generation = (gpd.GeoDataFrame(), gpd.GeoDataFrame())
    best_multipolygon_count = float("inf")
    best_error = float("inf")

    for i in range(attempts):
        try:
            attempt_seed = run_seed + i
            gen_points = _sample_points_from_global_pool(total_sites=total_sites, seed=attempt_seed)

            rng = np.random.default_rng(attempt_seed)
            point2zone = base_point2zone.copy()
            rng.shuffle(point2zone)

            generator_points_xy = gen_points.flatten().round(8).tolist()

            point_fixed_mask = [0.0 for _ in range(total_sites * 2)]
            point2zone_list = point2zone.tolist()

            for x, y, zone_idx in fixed_points:
                generator_points_xy.extend([round(x, 8), round(y, 8)])
                point_fixed_mask.extend([1.0, 1.0])
                point2zone_list.append(int(zone_idx))

            normalized_border = [round(v, 8) for xy in normalized_polygon.exterior.normalize().coords[::-1] for v in xy]

            vororoi_sites = optimize_territory_zoning(
                boundary_xy=normalized_border,
                generator_points_xy=generator_points_xy,
                point2zone=point2zone_list,
                point_fixed_mask=point_fixed_mask,
                zone_target_area=areas["area"].sort_index().round(8).tolist(),
                zone_neighbors=zone_neighbors,
                zone_forbidden=zone_forbidden,
                write_logs=write_logs,
                run_name=run_name,
            )

            site2idx = res[0]  # number of points [0,5,10,15,20] means there are 4 polygons with indexes 0..5 etc
            idx2vtxv = res[1]  # node indexes for each voronoi poly
            vtxv2xy = res[2]  # all points from generation (+bounds)
            edge2vtxv_wall = res[3]

            polygons, poly_sites = create_polygons(site2idx, site2room, idx2vtxv, np.array(vtxv2xy).flatten().tolist())
            devided_zones = gpd.GeoDataFrame(
                geometry=polygons, data=poly_sites, columns=["zone_id"], crs=local_crs
            ).dissolve("zone_id", as_index=False)

            # это ошибка генерации
            if len(devided_zones) != len(areas):
                raise ValueError(f"Number of devided_zones does not match {len(areas)}: {len(devided_zones)}")

            devided_zones = devided_zones.merge(areas.reset_index(), left_on="zone_id", right_on="index")

            multipolygon_count = sum(isinstance(geom, MultiPolygon) for geom in devided_zones.geometry)

            new_roads = [
                (vtxv2xy[x[0]], vtxv2xy[x[1]])
                for x in np.array(edge2vtxv_wall).reshape(int(len(edge2vtxv_wall) / 2), 2)
            ]
            new_roads = gpd.GeoDataFrame(geometry=[LineString(x) for x in new_roads], crs=local_crs)

            # Если мультиполигон
            if multipolygon_count > 0:

                # если разрешены мультиполигоны — отдаём как есть
                devided_out = devided_zones.drop(
                    columns=["zone_id", "index", "ratio", "area", "ratio_sqrt", "area_sqrt", "site_indeed"]
                )
                if allow_multipolygon:
                    return devided_out.explode(ignore_index=True), new_roads

                # иначе: оцениваем и сохраняем лучший
                actual_areas = devided_zones.geometry.area
                target_areas = devided_zones["area"]
                area_error = np.mean(np.abs(actual_areas - target_areas))

                if multipolygon_count < best_multipolygon_count or (
                    multipolygon_count == best_multipolygon_count and area_error < best_error
                ):
                    best_generation = (devided_out.copy(), new_roads.copy())
                    best_multipolygon_count = multipolygon_count
                    best_error = area_error

                raise MultiPolygonSplitError("MultiPolygon returned from optimizer. Recalculating.")

            devided_zones = devided_zones.drop(
                columns=["zone_id", "index", "ratio", "area", "ratio_sqrt", "area_sqrt", "site_indeed"]
            )
            return devided_zones, new_roads

        except MultiPolygonSplitError as e:
            if write_logs:
                print(e)
            continue

        except Exception as e:
            if write_logs:
                print(e)
            continue

    devided_zones, new_roads = best_generation
    if len(devided_zones) > 0:
        return devided_zones.explode(ignore_index=True), new_roads

    devided_zones = gpd.GeoDataFrame(geometry=[polygon_to_split], crs=local_crs)
    devided_zones["zone_name"] = fallback_zone_name

    new_roads = gpd.GeoDataFrame(geometry=[], crs=local_crs)
    return devided_zones, new_roads
