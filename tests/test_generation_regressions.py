"""Regression checks for geometry loss, duplicate inputs and bounded work."""

import queue
import time

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Polygon, box

from genplanner import FunctionalZone, GenPlanner, TerritoryZone
from genplanner._rust import optimize_territory_zoning
from genplanner.main.genplanner import split_queue
from genplanner.main.init_validation import cut_by_roads
from genplanner.tasks import feat2blocks
from genplanner.tasks.polygon_splitter import split_polygon
from genplanner.utils import territory_splitter
from genplanner.zones import BasicZone, TerritoryZoneKind


def test_block_geometries_keep_position_when_clip_indices_have_gaps(monkeypatch):
    zone = TerritoryZone(kind=TerritoryZoneKind.RESIDENTIAL, name="residential", min_block_area=1)
    split_blocks = gpd.GeoDataFrame(
        geometry=gpd.GeoSeries([box(0, 0, 1, 1), box(1, 0, 2, 1)], index=[1, 2]), crs=3857
    )
    split_blocks["zone"] = ["a", "b"]
    monkeypatch.setattr(feat2blocks, "split_polygon", lambda **kwargs: (split_blocks, gpd.GeoDataFrame()))

    result = feat2blocks.feature2blocks_splitter(
        (box(0, 0, 10, 10), [2], 1, 1, [10]),
        local_crs=3857,
        territory_zone=zone,
        simplify=0.01,
        rust_write_logs=False,
        run_name="index_regression",
    )

    assert result["generation"].geometry.notna().all()
    assert result["generation"].geometry.to_list() == split_blocks.geometry.to_list()


def test_duplicate_roads_keep_one_geometry_and_the_largest_width():
    territory = gpd.GeoDataFrame(geometry=[box(0, 0, 100, 100)], crs=3857)
    road = LineString([(-10, 50), (110, 50)])
    roads = gpd.GeoDataFrame({"roads_width": [5, 7]}, geometry=[road, road], crs=3857)

    _, generated_roads = cut_by_roads(territory, roads)

    assert generated_roads.geometry.to_wkb().duplicated().sum() == 0
    assert set(generated_roads["roads_width"]) == {7}


def test_point_selected_territory_faces_match_clipped_faces():
    territory = gpd.GeoDataFrame(
        {"zone": ["a", "b"]},
        geometry=[
            Polygon(box(0, 0, 1000, 1000).exterior, holes=[box(200, 200, 300, 300).exterior]),
            box(1000, 0, 2000, 1000),
        ],
        crs=3857,
    )
    roads = gpd.GeoDataFrame(
        geometry=[LineString([(-100, 500), (2100, 500)]).buffer(15)], crs=3857
    )

    clipped = territory_splitter(territory, roads, reproject_attr=True)
    selected = territory_splitter(territory, roads, reproject_attr=True, select_by_point=True)

    clipped_shapes = sorted((zone, geom.normalize().wkb) for zone, geom in zip(clipped.zone, clipped.geometry))
    selected_shapes = sorted((zone, geom.normalize().wkb) for zone, geom in zip(selected.zone, selected.geometry))
    assert selected_shapes == clipped_shapes


def test_expired_generation_budget_fails_with_task_count():
    task_queue = queue.Queue()
    with pytest.raises(TimeoutError, match="after 0 tasks"):
        split_queue(task_queue, local_crs=3857, parallel=False, max_workers=1, deadline=time.monotonic() - 1)


def test_generation_limits_require_positive_values():
    territory = gpd.GeoDataFrame(geometry=[box(0, 0, 100, 100)], crs=3857)
    with pytest.raises(ValueError, match="max_run_seconds"):
        GenPlanner(territory, max_run_seconds=0)
    with pytest.raises(ValueError, match="max_optimization_iterations"):
        GenPlanner(territory, max_optimization_iterations=0)


def test_parallel_generation_uses_bounded_optimizer():
    territory = gpd.GeoDataFrame(geometry=[box(0, 0, 1000, 1000)], crs=3857)
    funczone = FunctionalZone(
        {
            TerritoryZone(kind=TerritoryZoneKind.RESIDENTIAL, name="a", min_block_area=100000): 0.5,
            TerritoryZone(kind=TerritoryZoneKind.BUSINESS, name="b", min_block_area=100000): 0.5,
        },
        name="parallel_regression",
    )
    planner = GenPlanner(
        territory,
        parallel=True,
        parallel_max_workers=2,
        max_run_seconds=20,
        max_optimization_iterations=100,
    )

    zones, _ = planner.features2terr_zones2blocks(funczone=funczone, relation_matrix="empty")

    assert len(zones) > 0
    assert zones.geometry.notna().all()


def test_voronoi_split_covers_polygon_with_valid_zones():
    polygon = box(0, 0, 1000, 1000)
    zones = [BasicZone("a"), BasicZone("b"), BasicZone("c")]

    generated, _ = split_polygon(
        polygon_to_split=polygon,
        zone_ratios={zone: 1 / 3 for zone in zones},
        zone_neighbors=[],
        zone_forbidden=[],
        zone_fixed_point={},
        local_crs=3857,
        run_name="coverage_regression",
        geom_simplify_tol=0.01,
        seed=7,
        max_iterations=200,
    )

    assert set(generated["zone"]) == set(zones)
    assert generated.geometry.is_valid.all()
    assert polygon.difference(generated.union_all()).area < 1e-6


@pytest.mark.parametrize(
    ("boundary", "sites", "message"),
    [
        ([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0], [0.25, 0.5, 0.25, 0.5], "Coincident"),
        (
            [0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            [0.25, 0.5, 0.75, 0.5],
            "bisector passes through",
        ),
    ],
)
def test_degenerate_voronoi_inputs_return_errors_without_panicking(boundary, sites, message):
    with pytest.raises(RuntimeError, match=message):
        optimize_territory_zoning(
            boundary_xy=boundary,
            generator_points_xy=sites,
            point2zone=[0, 1],
            point_fixed_mask=[0.0] * 4,
            zone_target_area=[0.5, 0.5],
            zone_neighbors=[],
            zone_forbidden=[],
            write_logs=False,
            run_name="degenerate_test",
            max_iterations=10,
        )
