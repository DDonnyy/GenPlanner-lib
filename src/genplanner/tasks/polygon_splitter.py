import json
import math

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS
from shapely import Point, linestrings, points, voronoi_polygons
from shapely.geometry import MultiPoint, MultiPolygon, Polygon

from genplanner._config import config
from genplanner._rust import optimize_territory_zoning
from genplanner.utils import (
    denormalize_coords,
    normalize_coords,
    polygon_angle,
    rotate_coords,
)
from genplanner.zoning.abc_zone import Zone

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
        if not isinstance(k, Zone):
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

    seen_undirected: set[frozenset["Zone"]] = set()

    for i, item in enumerate(pairs):
        if not isinstance(item, tuple) or len(item) != 2:
            raise SplitPolygonValidationError(
                field,
                f"item #{i} must be tuple(BaseZone, BaseZone), got {item!r}",
            )
        a, b = item
        if not isinstance(a, Zone) or not isinstance(b, Zone):
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
        if not isinstance(k, Zone):
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
    zone_ratios: dict[Zone, float],
    zone_neighbors: list[tuple[Zone, Zone]],
    zone_forbidden: list[tuple[Zone, Zone]],
    zone_fixed_point: dict[Zone, Point],
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


def _allocate_sites_ratio(
    zone_ratios: dict["Zone", float],
    total_sites: int,
    min_per_zone: int = 2,
) -> dict["Zone", int]:
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

    w = ratios / ratios.sum()

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
    zone_ratios: dict[Zone, float],
    zone_neighbors: list[tuple[Zone, Zone]],
    zone_forbidden: list[tuple[Zone, Zone]],
    zone_fixed_point: dict[Zone, Point],
    local_crs: CRS,
    run_name: str,
    normalize_rotation: bool = True,
    allow_multipolygon=False,
    write_logs=False,
    seed=None,
    sites_multiplier=5,
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

    zones_ordered = list(areas_init["zone"])
    zone2idx = {z: i for i, z in enumerate(zones_ordered)}
    idx2zone = {i: z for i, z in enumerate(zones_ordered)}

    zone_neighbors_idx: list[tuple[int, int]] = [(zone2idx[a], zone2idx[b]) for (a, b) in zone_neighbors]
    zone_forbidden_idx: list[tuple[int, int]] = [(zone2idx[a], zone2idx[b]) for (a, b) in zone_forbidden]

    pivot_point = None
    angle_rad2rotate = None

    normalized_polygon = polygon_to_split
    if normalize_rotation:
        pivot_point = normalized_polygon.centroid
        angle_rad2rotate = polygon_angle(normalized_polygon)
        normalized_polygon = Polygon(rotate_coords(normalized_polygon.exterior.coords, pivot_point, -angle_rad2rotate))

    bounds = normalized_polygon.bounds

    # TODO add simplify
    normalized_polygon = Polygon(normalize_coords(normalized_polygon.exterior.coords, bounds))
    normalized_border = [round(v, 8) for xy in normalized_polygon.exterior.normalize().coords[::-1] for v in xy]

    fixed_points = []
    for zone, point in zone_fixed_point.items():
        room_idx = zone2idx[zone]
        if normalize_rotation:
            point = Point(rotate_coords(point.coords, pivot_point, -angle_rad2rotate))
        xy = normalize_coords(point.coords, bounds)
        fixed_points.append((xy[0][0], xy[0][1], room_idx))

    full_area = normalized_polygon.area
    areas = areas_init.copy()
    areas["ratio"] = areas["ratio"] / (areas["ratio"].sum())
    areas["area"] = areas["ratio"] * full_area

    Z = len(zone_ratios)
    total_sites = sites_multiplier * Z

    zone2count = _allocate_sites_ratio(zone_ratios=zone_ratios, total_sites=total_sites, min_per_zone=2)
    counts_by_idx = np.zeros(Z, dtype=np.int64)
    for z, cnt in zone2count.items():
        counts_by_idx[zone2idx[z]] = cnt
    base_point2zone = np.repeat(np.arange(Z, dtype=np.int64), counts_by_idx)

    if seed is not None:
        if not isinstance(seed, int):
            raise ValueError("seed must be int or None")
        run_seed = seed & 0xFFFFFFFF
    else:
        run_seed = np.random.SeedSequence().entropy
        run_seed = int(run_seed) & 0xFFFFFFFF

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

            if write_logs:
                run_name = f"{run_name}"
                log_path = f"{run_name}.jsonl"
                meta = {
                    "type": "meta",
                    "seed": int(run_seed),
                    "crs": str(local_crs),
                    "normalize_rotation": bool(normalize_rotation),
                    "allow_multipolygon": bool(allow_multipolygon),
                    "polygon_coords": normalized_border,
                }
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write(json.dumps(meta, ensure_ascii=False) + "\n")

            voronoi_coords, roads_coords = optimize_territory_zoning(
                boundary_xy=normalized_border,
                generator_points_xy=generator_points_xy,
                point2zone=point2zone_list,
                point_fixed_mask=point_fixed_mask,
                zone_target_area=areas["area"].sort_index().round(8).tolist(),
                zone_neighbors=zone_neighbors_idx,
                zone_forbidden=zone_forbidden_idx,
                write_logs=write_logs,
                run_name=run_name,
            )

            voronoi_coords = np.asarray(voronoi_coords, dtype=np.float64).reshape(-1, 2)
            voronoi_coords = denormalize_coords(voronoi_coords, bounds)  # shape (n, 2)

            if normalize_rotation:
                voronoi_coords = rotate_coords(voronoi_coords, pivot_point, +angle_rad2rotate)

            voronoi_points = points(voronoi_coords)
            voronoi_polys = list(voronoi_polygons(MultiPoint(voronoi_points)).geoms)

            voronoi_points_gdf = gpd.GeoDataFrame({"zone_id": point2zone_list}, geometry=voronoi_points, crs=local_crs)
            voronoi_polys_gdf = gpd.GeoDataFrame(geometry=voronoi_polys, crs=local_crs)

            voronoi_polys_gdf = voronoi_polys_gdf.sjoin(voronoi_points_gdf, how="left", predicate="contains")
            zones_gdf = voronoi_polys_gdf.dissolve(by="zone_id", as_index=False)

            zones_gdf = zones_gdf.clip(polygon_to_split, keep_geom_type=True)
            zones_gdf["zone"] = zones_gdf["zone_id"].map(idx2zone)

            multipolygon_count = sum(isinstance(geom, MultiPolygon) for geom in zones_gdf.geometry)

            # roads
            roads_arr = np.asarray(roads_coords, dtype=np.float64).reshape(-1, 4)
            p0 = roads_arr[:, 0:2]
            p1 = roads_arr[:, 2:4]
            p0 = denormalize_coords(p0, bounds)
            p1 = denormalize_coords(p1, bounds)
            if normalize_rotation:
                p0 = rotate_coords(p0, pivot_point, +angle_rad2rotate)
                p1 = rotate_coords(p1, pivot_point, +angle_rad2rotate)
            line_coords = np.stack([p0, p1], axis=1)
            road_geoms = linestrings(line_coords)
            roads_gdf = gpd.GeoDataFrame(geometry=road_geoms, crs=local_crs)

            if multipolygon_count > 0:
                if allow_multipolygon:
                    return zones_gdf[["zone", "geometry"]], roads_gdf

                actual_areas = zones_gdf.set_index("zone_id").geometry.area
                target_areas = areas.set_index(areas.index)["area"]
                aligned = actual_areas.reindex(target_areas.index)
                if aligned.isna().any():
                    area_error = float("inf")
                else:
                    area_error = float(np.mean(np.abs(aligned.values - target_areas.values)))

                if (multipolygon_count < best_multipolygon_count) or (
                    multipolygon_count == best_multipolygon_count and area_error < best_error
                ):
                    best_generation = (zones_gdf.copy(), roads_gdf.copy())
                    best_multipolygon_count = multipolygon_count
                    best_error = area_error

                raise MultiPolygonSplitError(
                    f"MultiPolygon returned (count={multipolygon_count}, area_error={area_error:.6f}). Recalculating."
                )

            return zones_gdf[["zone", "geometry"]], roads_gdf

        except MultiPolygonSplitError as e:
            continue

        except Exception as e:
            print(e)
            continue

    best_zones, best_roads = best_generation
    if len(best_zones) > 0:
        return best_zones[["zone", "geometry"]], best_roads

    fallback = gpd.GeoDataFrame(geometry=[polygon_to_split], crs=local_crs)
    fallback["zone"] = fallback_zone_name
    empty_roads = gpd.GeoDataFrame()
    return fallback, empty_roads
