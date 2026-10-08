"""Regression checks for geometry loss, duplicate inputs and bounded work."""

import importlib
import queue
import time

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Point, Polygon, box

from genplanner import FunctionalZone, GenPlanner, TerritoryZone
from genplanner._rust import optimize_territory_zoning
from genplanner.errors import FixPointsOutsideTerritoryError, GenPlannerArgumentError, SplitPolygonValidationError
from genplanner.main.genplanner import split_queue
from genplanner.main.init_validation import cut_by_roads, prepare_fixed_points_and_balance_ratios
from genplanner.tasks import feat2blocks
from genplanner.tasks.polygon_splitter import split_polygon
from genplanner.utils import elastic_wrap, territory_splitter
from genplanner.zones import BasicZone, TerritoryZoneKind


def _delayed_parallel_task(task, **kwargs):
    time.sleep(0.05 if task == 0 else 0.005)
    generated = gpd.GeoDataFrame({"task": [task]}, geometry=[box(task * 10, 0, task * 10 + 5, 5)], crs=3857)
    return {"generation": generated}


def test_block_geometries_keep_position_when_clip_indices_have_gaps(monkeypatch):
    zone = TerritoryZone(kind=TerritoryZoneKind.RESIDENTIAL, name="residential", min_block_area=1)
    split_blocks = gpd.GeoDataFrame(geometry=gpd.GeoSeries([box(0, 0, 1, 1), box(1, 0, 2, 1)], index=[1, 2]), crs=3857)
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
    roads = gpd.GeoDataFrame(geometry=[LineString([(-100, 500), (2100, 500)]).buffer(15)], crs=3857)

    clipped = territory_splitter(territory, roads, reproject_attr=True)
    selected = territory_splitter(territory, roads, reproject_attr=True, select_by_point=True)

    clipped_shapes = sorted((zone, geom.normalize().wkb) for zone, geom in zip(clipped.zone, clipped.geometry))
    selected_shapes = sorted((zone, geom.normalize().wkb) for zone, geom in zip(selected.zone, selected.geometry))
    assert selected_shapes == clipped_shapes


def test_known_working_crs_matches_estimated_crs_for_local_territory():
    territory = gpd.GeoDataFrame(
        {"zone": ["a", "b"]},
        geometry=[box(500000, 6000000, 501000, 6001000), box(501000, 6000000, 502000, 6001000)],
        crs=32631,
    )
    roads = gpd.GeoDataFrame(geometry=[LineString([(499900, 6000500), (502100, 6000500)]).buffer(5)], crs=32631)

    estimated = territory_splitter(territory, roads, reproject_attr=True, select_by_point=True)
    provided = territory_splitter(territory, roads, reproject_attr=True, select_by_point=True, working_crs=32631)

    assert estimated["zone"].tolist() == provided["zone"].tolist()
    assert estimated.geometry.normalize().to_wkb().tolist() == provided.geometry.normalize().to_wkb().tolist()


def test_expired_generation_budget_fails_with_task_count():
    task_queue = queue.Queue()
    with pytest.raises(TimeoutError, match="after 0 tasks"):
        split_queue(task_queue, local_crs=3857, parallel=False, max_workers=1, deadline=time.monotonic() - 1)


def test_parallel_queue_keeps_input_order_when_tasks_finish_out_of_order():
    task_queue = queue.Queue()
    for task in range(3):
        task_queue.put((_delayed_parallel_task, task, {}))

    generated, roads = split_queue(task_queue, local_crs=3857, parallel=True, max_workers=2)

    assert generated["task"].tolist() == [0, 1, 2]
    assert roads.empty
    assert roads.crs == generated.crs


def test_generation_limits_require_positive_values():
    territory = gpd.GeoDataFrame(geometry=[box(0, 0, 100, 100)], crs=3857)
    with pytest.raises(ValueError, match="max_run_seconds"):
        GenPlanner(territory, max_run_seconds=0)
    with pytest.raises(ValueError, match="max_optimization_iterations"):
        GenPlanner(territory, max_optimization_iterations=0)
    with pytest.raises(ValueError, match="seed"):
        GenPlanner(territory, seed=True)


def test_seeded_planner_repeats_zone_and_road_geometries():
    territory = gpd.GeoDataFrame(geometry=[box(0, 0, 1000, 1000)], crs=3857)
    funczone = FunctionalZone(
        {
            TerritoryZone(kind=TerritoryZoneKind.RESIDENTIAL, name="a", min_block_area=100000): 0.5,
            TerritoryZone(kind=TerritoryZoneKind.BUSINESS, name="b", min_block_area=100000): 0.5,
        },
        name="seed_regression",
    )

    def generate():
        planner = GenPlanner(territory, parallel=False, seed=7, max_optimization_iterations=100)
        return planner.features2terr_zones(funczone=funczone, relation_matrix="empty")

    (first_zones, first_roads), (second_zones, second_roads) = generate(), generate()

    assert len(first_zones) >= 2
    assert len(first_roads) > 0
    assert first_zones["territory_zone"].tolist() == second_zones["territory_zone"].tolist()
    assert first_zones.geometry.normalize().to_wkb().tolist() == second_zones.geometry.normalize().to_wkb().tolist()
    assert first_roads.geometry.normalize().to_wkb().tolist() == second_roads.geometry.normalize().to_wkb().tolist()
    assert first_zones.union_all().symmetric_difference(second_zones.union_all()).area == 0


def test_seeded_multi_feature_generation_handles_no_roads():
    territory = gpd.GeoDataFrame(geometry=[box(0, 0, 1000, 1000), box(1000, 0, 2000, 1000)], crs=3857)
    funczone = FunctionalZone(
        {
            TerritoryZone(kind=TerritoryZoneKind.RESIDENTIAL, name="a", min_block_area=100000): 0.5,
            TerritoryZone(kind=TerritoryZoneKind.BUSINESS, name="b", min_block_area=100000): 0.5,
        },
        name="multi_seed_regression",
    )
    planner = GenPlanner(territory, parallel=False, seed=7, max_optimization_iterations=100)

    first_zones, first_roads = planner.features2terr_zones(funczone=funczone, relation_matrix="empty")
    second_zones, second_roads = planner.features2terr_zones(funczone=funczone, relation_matrix="empty")

    assert len(first_zones) == 2
    assert len(first_roads) == len(second_roads) == 0
    assert first_roads.crs == territory.crs
    assert first_zones.geometry.normalize().to_wkb().tolist() == second_zones.geometry.normalize().to_wkb().tolist()
    assert first_zones.union_all().symmetric_difference(territory.union_all()).area < 1e-6


def test_seeded_multi_feature_generation_repeats_split_geometry():
    territory = gpd.GeoDataFrame(
        geometry=[
            box(0, 0, 500, 500),
            box(600, 0, 1100, 500),
            box(0, 600, 500, 1100),
            box(600, 600, 1100, 1100),
        ],
        crs=3857,
    )
    funczone = FunctionalZone(
        {
            TerritoryZone(kind=TerritoryZoneKind.RESIDENTIAL, name="a", min_block_area=10000): 0.3,
            TerritoryZone(kind=TerritoryZoneKind.BUSINESS, name="b", min_block_area=10000): 0.3,
            TerritoryZone(kind=TerritoryZoneKind.RECREATION, name="c", min_block_area=10000): 0.4,
        },
        name="multi_split_seed_regression",
    )
    planner = GenPlanner(territory, parallel=False, seed=7, max_optimization_iterations=100)

    first_zones, first_roads = planner.features2terr_zones(funczone=funczone, relation_matrix="empty")
    second_zones, second_roads = planner.features2terr_zones(funczone=funczone, relation_matrix="empty")

    assert len(first_zones) > len(territory)
    assert len(first_roads) > 0
    assert first_zones["territory_zone"].tolist() == second_zones["territory_zone"].tolist()
    assert first_zones.geometry.normalize().to_wkb().tolist() == second_zones.geometry.normalize().to_wkb().tolist()
    assert first_roads.geometry.normalize().to_wkb().tolist() == second_roads.geometry.normalize().to_wkb().tolist()


def test_elastic_wrap_keeps_the_same_envelope_for_separated_features():
    features = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10), box(20, 0, 30, 10), box(60, 0, 70, 10)], crs=3857)
    components = gpd.GeoDataFrame(geometry=[features.union_all()], crs=features.crs).explode(ignore_index=True)
    nearest_distances = components.apply(lambda row: components.drop(row.name).distance(row.geometry).min(), axis=1)
    old_radius = (np.ceil(nearest_distances.max()) + 0.1) * 1.1
    expected = components.buffer(old_radius + 1, quad_segs=2).union_all().buffer(-old_radius, quad_segs=2)

    assert elastic_wrap(features).equals_exact(expected, 0)


def test_existing_zone_balancing_removes_satisfied_zone_and_its_fixed_point():
    residential = TerritoryZone(kind=TerritoryZoneKind.RESIDENTIAL, name="residential")
    business = TerritoryZone(kind=TerritoryZoneKind.BUSINESS, name="business")
    territory = gpd.GeoDataFrame(geometry=[box(0, 0, 100, 100)], crs=3857)
    existing = gpd.GeoDataFrame({"territory_zone": [residential]}, geometry=[box(0, 0, 50, 100)], crs=3857)
    user_points = gpd.GeoDataFrame(
        {"fixed_zone": [residential, business]}, geometry=[Point(75, 50), Point(60, 50)], crs=3857
    )
    static_points = gpd.GeoDataFrame({"fixed_zone": [business]}, geometry=[Point(25, 50)], crs=3857)

    points, ratios = prepare_fixed_points_and_balance_ratios(
        {residential: 0.25, business: 0.75}, user_points, static_points, territory, existing
    )

    assert ratios == {business: 0.75}
    assert points["fixed_zone"].tolist() == [business]
    assert points.geometry.to_list() == [Point(60, 50)]


def test_fixed_point_outside_territory_has_contextual_error():
    territory = gpd.GeoDataFrame(geometry=[box(0, 0, 100, 100)], crs=3857)
    zone = TerritoryZone(kind=TerritoryZoneKind.RESIDENTIAL, name="residential")
    points = gpd.GeoDataFrame({"fixed_zone": [zone]}, geometry=[Point(200, 200)], crs=3857)
    planner = GenPlanner(territory, parallel=False)

    with pytest.raises(FixPointsOutsideTerritoryError, match="outside the working territory"):
        planner.features2terr_zones(FunctionalZone({zone: 1}, name="fixed_point"), terr_zones_fix_points=points)


def test_fixed_point_is_reprojected_before_territory_check():
    territory = gpd.GeoDataFrame(geometry=[box(0, 0, 100, 100)], crs=3857)
    zone = TerritoryZone(kind=TerritoryZoneKind.RESIDENTIAL, name="residential")
    points = gpd.GeoDataFrame({"fixed_zone": [zone]}, geometry=[Point(50, 50)], crs=3857)
    planner = GenPlanner(territory, parallel=False)

    prepared, _ = prepare_fixed_points_and_balance_ratios(
        {zone: 1}, points, planner.static_fix_points, planner.territory_to_work_with, planner.existing_terr_zones
    )

    assert prepared.crs == planner.local_crs
    assert prepared.geometry.iloc[0].equals_exact(points.to_crs(planner.local_crs).geometry.iloc[0], 0)
    assert points.crs == territory.crs


def test_fixed_point_requires_known_crs():
    territory = gpd.GeoDataFrame(geometry=[box(0, 0, 100, 100)], crs=3857)
    zone = TerritoryZone(kind=TerritoryZoneKind.RESIDENTIAL, name="residential")
    points = gpd.GeoDataFrame({"fixed_zone": [zone]}, geometry=[Point(50, 50)])
    planner = GenPlanner(territory, parallel=False)

    with pytest.raises(GenPlannerArgumentError, match="must have a CRS"):
        prepare_fixed_points_and_balance_ratios(
            {zone: 1}, points, planner.static_fix_points, planner.territory_to_work_with, planner.existing_terr_zones
        )


def test_split_polygon_rejects_duplicate_undirected_zone_pair():
    a, b = BasicZone("a"), BasicZone("b")
    with pytest.raises(SplitPolygonValidationError, match="duplicate undirected pair"):
        split_polygon(
            polygon_to_split=box(0, 0, 100, 100),
            zone_ratios={a: 0.5, b: 0.5},
            zone_neighbors=[(a, b), (b, a)],
            zone_forbidden=[],
            zone_fixed_point={},
            local_crs=3857,
            run_name="invalid_relation",
            geom_simplify_tol=0.01,
        )


def test_planner_buffers_each_road_by_its_own_width(monkeypatch):
    territory = gpd.GeoDataFrame(geometry=[box(0, 0, 100, 100)], crs=3857)
    planner = GenPlanner(territory, parallel=False)
    zones = gpd.GeoDataFrame(geometry=[planner.territory_to_work_with.geometry.iloc[0]], crs=planner.local_crs)
    roads = gpd.GeoDataFrame(
        {"roads_width": [3, 9]},
        geometry=[LineString([(0, 20), (100, 20)]), LineString([(0, 80), (100, 80)])],
        crs=planner.local_crs,
    )
    captured = {}
    planner_module = importlib.import_module("genplanner.main.genplanner")
    monkeypatch.setattr(planner_module, "split_queue", lambda *args, **kwargs: (zones, roads))

    def capture_splitters(zones_to_split, splitters, **kwargs):
        captured["splitters"] = splitters
        return zones_to_split

    monkeypatch.setattr(planner_module, "territory_splitter", capture_splitters)

    planner._run(lambda task, **kwargs: None)

    expected = [geom.buffer(width / 2, quad_segs=4) for geom, width in zip(roads.geometry, roads.roads_width)]
    assert captured["splitters"].geometry.to_list() == expected


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


def test_seeded_parallel_block_generation_repeats_structure():
    territory = gpd.GeoDataFrame(geometry=[box(0, 0, 1000, 1000)], crs=3857)
    funczone = FunctionalZone(
        {
            TerritoryZone(kind=TerritoryZoneKind.RESIDENTIAL, name="a", min_block_area=100000): 0.5,
            TerritoryZone(kind=TerritoryZoneKind.BUSINESS, name="b", min_block_area=100000): 0.5,
        },
        name="parallel_seed_regression",
    )
    planner = GenPlanner(
        territory, parallel=True, parallel_max_workers=2, seed=7, max_run_seconds=20, max_optimization_iterations=100
    )

    (first_blocks, first_roads), (second_blocks, second_roads) = (
        planner.features2terr_zones2blocks(funczone=funczone, relation_matrix="empty") for _ in range(2)
    )

    assert len(first_blocks) > 2
    assert first_blocks["territory_zone"].tolist() == second_blocks["territory_zone"].tolist()
    assert first_blocks.geometry.normalize().to_wkb().tolist() == second_blocks.geometry.normalize().to_wkb().tolist()
    assert first_roads.geometry.normalize().to_wkb().tolist() == second_roads.geometry.normalize().to_wkb().tolist()


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


@pytest.mark.parametrize(
    ("fixed_mask", "forbidden"),
    [([0.0] * 6, []), ([1.0, 1.0, 0.0, 0.0, 0.0, 0.0], []), ([0.0] * 6, [(0, 1)])],
)
def test_native_optimizer_repeats_with_fixed_sites_and_forbidden_neighbors(fixed_mask, forbidden):
    arguments = dict(
        boundary_xy=[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        generator_points_xy=[0.2, 0.2, 0.75, 0.3, 0.45, 0.8],
        point2zone=[0, 1, 2],
        point_fixed_mask=fixed_mask,
        zone_target_area=[0.3, 0.3, 0.4],
        zone_neighbors=[],
        zone_forbidden=forbidden,
        write_logs=False,
        run_name="native_repeatability",
        max_iterations=60,
    )

    first = optimize_territory_zoning(**arguments)
    second = optimize_territory_zoning(**arguments)

    assert first == second
    assert len(first[0]) == 6
    assert all(np.isfinite(first[0]))
    assert all(len(edge) == 4 and all(np.isfinite(edge)) for edge in first[1])


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"generator_points_xy": [0.2, 0.2]}, "two coordinates per point2zone"),
        ({"point_fixed_mask": [0.0]}, "two flags per site"),
        ({"point2zone": [0, 9]}, "point2zone\\[1\\] contains invalid zone index"),
        ({"point2zone": [0, 2**64 - 1], "point_fixed_mask": [0.0, 0.0, 1.0, 1.0]}, "fixed site 1 has no zone"),
        ({"zone_neighbors": [(0, 9)]}, "zone_neighbors\\[0\\] contains an invalid zone index"),
    ],
)
def test_native_optimizer_reports_invalid_array_shapes_and_zone_indices(changes, message):
    arguments = dict(
        boundary_xy=[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        generator_points_xy=[0.2, 0.2, 0.75, 0.3],
        point2zone=[0, 1],
        point_fixed_mask=[0.0] * 4,
        zone_target_area=[0.5, 0.5],
        zone_neighbors=[],
        zone_forbidden=[],
        write_logs=False,
        run_name="invalid_native_input",
        max_iterations=10,
    )
    arguments.update(changes)

    with pytest.raises(RuntimeError, match=message):
        optimize_territory_zoning(**arguments)
