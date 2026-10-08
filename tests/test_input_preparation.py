"""Input preparation cases that affect the geometry passed to the optimizer."""

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point, box

from genplanner import FunctionalZone, TerritoryZone
from genplanner.errors import GenPlannerArgumentError, GenPlannerInitError, RelationMatrixError
from genplanner.main.init_validation import (
    add_static_fix_points,
    cut_by_existing_terr_zones,
    cut_by_roads,
    cut_out_features,
    prepare_fixed_points_and_balance_ratios,
    resolve_relation_matrix,
    roads_width_def,
)
from genplanner.zone_relations import ZoneRelationMatrix
from genplanner.zones import TerritoryZoneKind


@pytest.fixture
def zones():
    return (
        TerritoryZone(kind=TerritoryZoneKind.RESIDENTIAL, name="residential"),
        TerritoryZone(kind=TerritoryZoneKind.BUSINESS, name="business"),
    )


def test_duplicate_exclusions_remove_area_once_and_preserve_other_features():
    territory = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10), box(20, 0, 30, 10)], crs=3857)
    exclusion = box(3, 3, 7, 7)
    exclusions = gpd.GeoDataFrame(geometry=[exclusion, exclusion], crs=3857)

    result = cut_out_features(territory, exclusions, exclude_buffer=0)

    assert result.geometry.is_valid.all()
    assert result.union_all().symmetric_difference(territory.union_all().difference(exclusion)).area < 1e-6


def test_exclusions_reject_crs_mismatch():
    territory = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10)], crs=3857)
    exclusions = gpd.GeoDataFrame(geometry=[box(3, 3, 7, 7)], crs=4326)

    with pytest.raises(GenPlannerInitError, match="CRS mismatch"):
        cut_out_features(territory, exclusions, exclude_buffer=0)


def test_road_without_width_gets_default_and_splits_territory():
    territory = gpd.GeoDataFrame(geometry=[box(0, 0, 100, 100)], crs=3857)
    roads = gpd.GeoDataFrame(geometry=[LineString([(-10, 50), (110, 50)])], crs=3857)

    pieces, generated = cut_by_roads(territory, roads)

    assert len(pieces) == 2
    assert pieces.union_all().symmetric_difference(territory.union_all()).area / territory.union_all().area < 1e-6
    assert generated["roads_width"].tolist() == [roads_width_def["local road"]]
    assert generated["road_lvl"].tolist() == ["user_roads"]


def test_static_fixed_point_uses_largest_part_of_existing_zone(zones):
    territory = gpd.GeoDataFrame(geometry=[box(0, 0, 100, 100)], crs=3857)
    existing = gpd.GeoDataFrame(
        {"territory_zone": [zones[0], zones[0]]},
        geometry=[box(5, 5, 10, 10), box(60, 30, 90, 70)],
        crs=3857,
    )

    fixed = add_static_fix_points(territory, existing, merge_radius=1)

    assert fixed["fixed_zone"].tolist() == [zones[0]]
    assert fixed.geometry.iloc[0].x > 50
    assert fixed.geometry.iloc[0].distance(territory.geometry.iloc[0].buffer(-0.1).boundary) < 1e-8


def test_existing_zone_merge_preserves_total_territory_area(zones):
    territory = gpd.GeoDataFrame(geometry=[box(0, 0, 100, 100)], crs=3857)
    existing = gpd.GeoDataFrame({"territory_zone": [zones[0]]}, geometry=[box(0, 0, 40, 100)], crs=3857)

    remaining, merged = cut_by_existing_terr_zones(territory, existing, existing_tz_fill_ratio=0.2)

    assert merged["territory_zone"].tolist() == [zones[0]]
    assert remaining.union_all().intersection(merged.union_all()).area < 1e-9
    assert remaining.union_all().union(merged.union_all()).symmetric_difference(box(0, 0, 100, 100)).area < 1e-9


def test_relation_matrix_rejects_unknown_preset_and_missing_zone(zones):
    functional = FunctionalZone({zones[0]: 0.5, zones[1]: 0.5}, name="input_relations")

    with pytest.raises(RelationMatrixError, match="Unknown relation_matrix preset"):
        resolve_relation_matrix(functional, "unknown")
    with pytest.raises(RelationMatrixError, match="misses zones"):
        resolve_relation_matrix(functional, ZoneRelationMatrix.empty((zones[0],)))


def test_fixed_points_reject_unknown_zone_and_nonpoint(zones):
    territory = gpd.GeoDataFrame(geometry=[box(0, 0, 100, 100)], crs=3857)
    empty_existing = gpd.GeoDataFrame(geometry=[], crs=3857)
    unknown = TerritoryZone(kind=TerritoryZoneKind.RECREATION, name="unknown")
    cases = [
        (gpd.GeoDataFrame({"fixed_zone": [unknown]}, geometry=[Point(50, 50)], crs=3857), "not present"),
        (gpd.GeoDataFrame({"fixed_zone": [zones[0]]}, geometry=[box(40, 40, 60, 60)], crs=3857), "Point"),
    ]

    for points, message in cases:
        with pytest.raises(GenPlannerArgumentError, match=message):
            prepare_fixed_points_and_balance_ratios({zones[0]: 1.0}, points, empty_existing, territory, empty_existing)
