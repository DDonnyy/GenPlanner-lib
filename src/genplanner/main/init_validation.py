import geopandas as gpd
import pandas as pd

from shapely.ops import unary_union, nearest_points

from genplanner.errors import (
    GenPlannerInitError,
    RelationMatrixError,
    GenPlannerArgumentError,
    FixPointsOutsideTerritoryError,
)
from genplanner.utils import (
    explode_linestring,
    extend_linestring,
    geom2multilinestring,
    territory_splitter,
)
from genplanner.zones import FunctionalZone
from genplanner import config
from genplanner.zone_relations.forbidden_terr_kind import FORBIDDEN_NEIGHBORHOOD
from genplanner.zone_relations.relation_matrix import ZoneRelationMatrix

logger = config.logger

roads_width_def = config.roads_width_def.copy()


def cut_out_features(
    features_gdf: gpd.GeoDataFrame,
    exclude_gdf: gpd.GeoDataFrame,
    exclude_buffer: float,
) -> gpd.GeoDataFrame:
    if features_gdf.crs != exclude_gdf.crs:
        raise GenPlannerInitError(
            f"CRS mismatch between features_gdf({features_gdf.crs}) and exclude_gdf({exclude_gdf.crs})."
        )
    exclude_gdf = exclude_gdf.clip(features_gdf.total_bounds, keep_geom_type=True)
    exclude_gdf.geometry = exclude_gdf.geometry.buffer(exclude_buffer, resolution=2)
    exclude_gdf = gpd.GeoDataFrame(geometry=[exclude_gdf.union_all()], crs=exclude_gdf.crs)
    return territory_splitter(features_gdf, exclude_gdf, return_splitters=False).reset_index(drop=True)


def cut_by_roads(
    features_gdf: gpd.GeoDataFrame, roads_gdf: gpd.GeoDataFrame, roads_extend_distance: float = 5
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    if features_gdf.crs != roads_gdf.crs:
        raise GenPlannerInitError(
            f"CRS mismatch between features_gdf({features_gdf.crs}) and roads_gdf({roads_gdf.crs})."
        )
    roads_gdf = roads_gdf.explode(ignore_index=True)
    splitters_roads = roads_gdf.copy()
    splitters_roads.geometry = splitters_roads.geometry.normalize()
    splitters_roads = splitters_roads[~splitters_roads.geometry.duplicated(keep="first")]
    splitters_roads.geometry = splitters_roads.geometry.apply(extend_linestring, distance=roads_extend_distance)
    features_gdf = territory_splitter(features_gdf, splitters_roads, return_splitters=False).reset_index(drop=True)

    splitters_centroids = gpd.GeoDataFrame(
        geometry=pd.Series(
            features_gdf.geometry.apply(geom2multilinestring).explode().apply(explode_linestring)
        ).explode(ignore_index=True),
        crs=features_gdf.crs,
    )
    splitters_centroids.geometry = splitters_centroids.geometry.centroid.buffer(0.1, resolution=1)
    roads_gdf["new_geometry"] = roads_gdf.geometry.apply(geom2multilinestring).explode().apply(explode_linestring)
    roads_gdf = roads_gdf.explode(column="new_geometry", ignore_index=True)
    roads_gdf["geometry"] = roads_gdf["new_geometry"]
    roads_gdf.drop(columns=["new_geometry"], inplace=True)

    roads_gdf = roads_gdf.sjoin(splitters_centroids, how="inner", predicate="intersects").drop(columns=["index_right"])
    roads_gdf = roads_gdf[~roads_gdf.index.duplicated(keep="first")]
    local_road_width = roads_width_def.get("local road")
    if "roads_width" not in roads_gdf.columns:
        logger.warning(
            f"Column 'roads_width' missing in GeoDataFrame, filling it with default local road width {local_road_width}"
        )
        roads_gdf["roads_width"] = local_road_width
    roads_gdf["roads_width"] = roads_gdf["roads_width"].fillna(local_road_width)
    roads_gdf["road_lvl"] = "user_roads"

    return features_gdf, roads_gdf


def add_static_fix_points(
    features_gdf: gpd.GeoDataFrame, existing_terr_zones: gpd.GeoDataFrame, merge_radius: float
) -> gpd.GeoDataFrame:
    """
    For each existing territory zone, finds a representative point and snaps it to the nearest point on the boundary of the features_gdf.
    If there are multiple geometries for a territory zone, the largest one will create fix point.
    """
    if features_gdf.crs != existing_terr_zones.crs:
        raise GenPlannerInitError(
            f"CRS mismatch between features_gdf({features_gdf.crs}) and existing_terr_zones({existing_terr_zones.crs})."
        )

    target = features_gdf.geometry.buffer(-0.1, resolution=1)
    target_boundary = target.boundary.union_all()

    rows = []
    for tz, ez in existing_terr_zones.groupby("territory_zone", sort=False):
        geoms = ez.geometry

        if len(ez) == 1:
            rep_pt = geoms.iloc[0].representative_point()
        else:
            merged = geoms.buffer(merge_radius, resolution=1).union_all()
            merged = merged.buffer(-merge_radius, resolution=1)

            parts = gpd.GeoSeries([merged], crs=ez.crs).explode(ignore_index=True)
            parts = parts[parts.notna() & (~parts.is_empty)]

            largest_poly = parts.loc[parts.area.idxmax()]
            rep_pt = largest_poly.representative_point()

        snapped = nearest_points(rep_pt, target_boundary)[1]
        rows.append({"fixed_zone": tz, "geometry": snapped})

    return gpd.GeoDataFrame(rows, crs=existing_terr_zones.crs)


def cut_by_existing_terr_zones(
    features_gdf: gpd.GeoDataFrame, existing_terr_zones: gpd.GeoDataFrame, existing_tz_fill_ratio: float
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    existing_tz_fill_ratio - the minimum ratio of features_gdf area that should be covered by existing_terr_zones
    for it to be merged into existing_terr_zones instead of being split further. Value should be in range [0, 1]

    """
    if "territory_zone" not in existing_terr_zones.columns:
        raise GenPlannerInitError(
            "`territory_zone` column not found in GeoDataFrame, but existing_terr_zones was provided"
        )

    if features_gdf.crs != existing_terr_zones.crs:
        raise GenPlannerInitError(
            f"CRS mismatch between features_gdf({features_gdf.crs}) and existing_terr_zones({existing_terr_zones.crs})."
        )

    existing_terr_zones = existing_terr_zones.clip(features_gdf, keep_geom_type=True)

    splitted_territory = territory_splitter(features_gdf, existing_terr_zones, return_splitters=True).reset_index(
        drop=True
    )

    splitted_territory["existing_area"] = splitted_territory.area
    splitted_territory.geometry = splitted_territory.representative_point()
    splitted_territory = splitted_territory.sjoin(existing_terr_zones, how="left").rename(
        columns={"index_right": "existing_zone_index"}
    )
    splitted_territory = splitted_territory[~splitted_territory["territory_zone"].isna()]

    features_gdf["full_area"] = features_gdf.area
    potential = features_gdf.sjoin(splitted_territory, how="left")
    potential = potential[~potential["territory_zone"].isna()]
    potential["ratio"] = (potential["existing_area"] / potential["full_area"]).round(2)
    consistent_idx = potential.groupby(level=0)["territory_zone"].nunique().pipe(lambda s: s[s == 1].index)
    potential = potential.loc[consistent_idx]
    potential: gpd.GeoDataFrame = potential[(potential["ratio"] >= existing_tz_fill_ratio)]

    for ezi, group in potential.groupby("existing_zone_index"):
        base = existing_terr_zones.at[ezi, "geometry"]
        merged = unary_union([base, *group.geometry.dropna().to_list()])
        existing_terr_zones.at[ezi, "geometry"] = merged

    existing_terr_zones["area"] = existing_terr_zones.geometry.area

    existing_terr_zones = existing_terr_zones[["territory_zone", "geometry"]]
    features_gdf = territory_splitter(features_gdf, existing_terr_zones, return_splitters=False).reset_index(drop=True)
    return features_gdf, existing_terr_zones


def resolve_relation_matrix(
    funczone: FunctionalZone,
    relation_matrix,
) -> ZoneRelationMatrix:
    zones = tuple(funczone.zones_ratio.keys())

    if relation_matrix is None:
        relation_matrix = "default"

    if isinstance(relation_matrix, str):
        key = relation_matrix.strip().lower()
        if key == "empty":
            return ZoneRelationMatrix.empty(zones)
        if key == "default":
            return ZoneRelationMatrix.from_kind_forbidden(
                zones=zones,
                kind_forbidden=FORBIDDEN_NEIGHBORHOOD,
            )
        raise RelationMatrixError(
            f"Unknown relation_matrix preset: {relation_matrix!r}. "
            "Allowed: 'empty', 'default' or ZoneRelationMatrix."
        )

    if isinstance(relation_matrix, ZoneRelationMatrix):
        missing = set(zones) - set(relation_matrix.zones)
        extra = set(relation_matrix.zones) - set(zones)
        if missing:
            raise RelationMatrixError(f"relation_matrix misses zones: {[z.name for z in missing]}")
        if extra:
            logger.warning(f"relation_matrix has extra zones not in funczone: {[z.name for z in extra]}")
        return relation_matrix

    raise RelationMatrixError("relation_matrix must be 'empty' | 'default' | ZoneRelationMatrix | None")


def prepare_fixed_points_and_balance_ratios(
    zones_ratio_dict: dict,
    fix_points: gpd.GeoDataFrame | None,
    static_fix_points: gpd.GeoDataFrame | None,
    features2split: gpd.GeoDataFrame,
    existing_terr_zones: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, dict]:

    if fix_points is None:
        fix_points = gpd.GeoDataFrame(columns=["fixed_zone"])
    else:
        if "fixed_zone" not in fix_points.columns:
            raise GenPlannerArgumentError("Column 'fixed_zone' is missing in the fixed_points GeoDataFrame.")
        if not (fix_points.geom_type == "Point").all():
            raise GenPlannerArgumentError("All geometries in fixed_points must be of type 'Point'.")

    valid_zones_keys = set(zones_ratio_dict.keys())

    if len(static_fix_points) > 0:
        sfx = static_fix_points.copy()
        sfx = sfx[sfx["fixed_zone"].isin(valid_zones_keys)]
        if len(fix_points) > 0:
            existing_labels = set(fix_points["fixed_zone"])
            sfx = sfx[~sfx["fixed_zone"].isin(existing_labels)]
        if len(sfx) > 0:
            fix_points = pd.concat([sfx[["fixed_zone", "geometry"]], fix_points], ignore_index=True)

    fixed_zones_values = set(fix_points["fixed_zone"])
    invalid_zones = fixed_zones_values - valid_zones_keys

    if invalid_zones:
        raise GenPlannerArgumentError(
            f"The following fixed_zone values are not present in zones_ratio_dict: {invalid_zones}\n"
            f"Available keys in zones_ratio_dict: {valid_zones_keys}\n"
            f"Provided fixed_zone values: {fixed_zones_values}"
        )
    if len(fix_points) > 0:
        joined = gpd.sjoin(fix_points, features2split, how="left", predicate="within")
        if joined["index_right"].isna().any():
            raise FixPointsOutsideTerritoryError(
                "Some points in fixed_zones are located outside the working territory geometries."
            )

    if len(existing_terr_zones) > 0:
        pieces = list(features2split.geometry) + list(existing_terr_zones.geometry)
        total_area = unary_union(pieces).area
        existing_ratios_by_zone: dict[str, float] = {}
        dissolved = existing_terr_zones.dissolve(by="territory_zone", as_index=False)
        dissolved["__area__"] = dissolved.geometry.area
        for _, row in dissolved.iterrows():
            z = row["territory_zone"]
            a = float(row["__area__"])
            existing_ratios_by_zone[z] = a / total_area
        balanced_ratio_dict = {}
        zero_ratio_zones = set()
        for z, target in zones_ratio_dict.items():
            existed = existing_ratios_by_zone.get(z, 0.0)
            remaining = max(float(target) - float(existed), 0.0)

            if remaining > 0:
                balanced_ratio_dict[z] = remaining
            else:
                zero_ratio_zones.add(z)

        zones_ratio_dict = balanced_ratio_dict
        if len(zero_ratio_zones) > 0 and len(fix_points) > 0:
            fix_points = fix_points[~fix_points["fixed_zone"].isin(zero_ratio_zones)].copy()

    return fix_points, zones_ratio_dict
