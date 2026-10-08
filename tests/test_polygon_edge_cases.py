"""Small offline tests for uncommon polygon subdivision paths."""

import importlib

import geopandas as gpd
import pytest
from shapely import points
from shapely.geometry import MultiPolygon, Point, Polygon, box

from genplanner.errors import SplitPolygonValidationError
from genplanner.tasks.polygon_splitter import _clip_zones, _validate_polygon, _zones_from_voronoi, split_polygon
from genplanner.zones import BasicZone


def _split_args(polygon, zones):
    return dict(
        polygon_to_split=polygon,
        zone_ratios={zones[0]: 0.6, zones[1]: 0.4},
        zone_neighbors=[],
        zone_forbidden=[],
        zone_fixed_point={},
        local_crs=3857,
        run_name="polygon_edge_case",
        geom_simplify_tol=0.01,
        seed=7,
        max_iterations=100,
    )


def test_polygon_with_hole_gets_connected_boundary_and_road():
    polygon = Polygon(box(0, 0, 100, 100).exterior, holes=[box(40, 40, 60, 60).exterior])

    connected, roads = _validate_polygon(polygon, run_name="hole_regression")

    assert connected.is_valid
    assert not connected.interiors
    assert len(roads) == 1
    assert roads[0].length > 0


def test_voronoi_grouping_matches_spatial_join_and_dissolve():
    polygons = [box(0, 0, 2, 2), box(1, 0, 3, 2), box(5, 5, 6, 6)]
    sites = points([(0.5, 1), (1.5, 1), (2.5, 1)])
    zone_ids = [0, 1, 1]
    polygon_frame = gpd.GeoDataFrame(geometry=polygons, crs=3857)
    point_frame = gpd.GeoDataFrame({"zone_id": zone_ids}, geometry=sites, crs=3857)
    reference = polygon_frame.sjoin(point_frame, how="left", predicate="contains").dissolve(
        by="zone_id", as_index=False
    )

    result = _zones_from_voronoi(polygons, sites, zone_ids, 3857)

    assert result["zone_id"].tolist() == reference["zone_id"].tolist()
    assert result.geometry.normalize().to_wkb().tolist() == reference.geometry.normalize().to_wkb().tolist()


def test_fast_polygon_clip_preserves_geopandas_row_order_and_geometry():
    zones = gpd.GeoDataFrame(
        {"zone_id": [0, 1, 2]},
        geometry=[box(0, 0, 5, 5), box(4, 0, 9, 5), box(20, 0, 25, 5)],
        crs=3857,
    )
    clip_polygon = box(2, 1, 7, 4)
    reference = zones.clip(clip_polygon, keep_geom_type=True)

    result = _clip_zones(zones, clip_polygon)

    assert result["zone_id"].tolist() == reference["zone_id"].tolist()
    assert result.geometry.normalize().to_wkb().tolist() == reference.geometry.normalize().to_wkb().tolist()


def test_native_errors_exhaust_attempts_then_return_one_valid_fallback_zone(monkeypatch):
    zones = (BasicZone("a"), BasicZone("b"))
    polygon = box(0, 0, 100, 100)
    calls = []

    def fail_optimizer(**kwargs):
        calls.append(kwargs["run_name"])
        raise RuntimeError("simulated native failure")

    module = importlib.import_module("genplanner.tasks.polygon_splitter")
    monkeypatch.setattr(module, "optimize_territory_zoning", fail_optimizer)

    generated, roads = split_polygon(**_split_args(polygon, zones))

    assert len(calls) == 5
    assert generated["zone"].tolist() == [zones[0]]
    assert generated.geometry.iloc[0].equals(polygon)
    assert roads.empty


def test_rejected_multipolygon_attempts_do_not_build_unused_roads(monkeypatch):
    zones = (BasicZone("a"), BasicZone("b"))
    module = importlib.import_module("genplanner.tasks.polygon_splitter")
    generated_zone_geometry = gpd.GeoDataFrame(
        {"zone_id": [0, 1]},
        geometry=[MultiPolygon([box(0, 0, 20, 40), box(0, 60, 20, 100)]), box(30, 0, 100, 100)],
        crs=3857,
    )
    calls = {"optimizer": 0, "roads": 0}

    def optimizer(**_kwargs):
        calls["optimizer"] += 1
        return [0.2, 0.2, 0.8, 0.8], [[0.1, 0.1, 0.9, 0.9]]

    real_linestrings = module.linestrings

    def track_roads(*args, **kwargs):
        calls["roads"] += 1
        return real_linestrings(*args, **kwargs)

    monkeypatch.setattr(module, "optimize_territory_zoning", optimizer)
    monkeypatch.setattr(module, "_zones_from_voronoi", lambda *_args: generated_zone_geometry.copy())
    monkeypatch.setattr(module, "linestrings", track_roads)

    generated, roads = split_polygon(**_split_args(box(0, 0, 100, 100), zones))

    assert calls == {"optimizer": 5, "roads": 1}
    assert len(generated) == 3
    assert len(roads) == 1


def test_irregular_polygon_with_fixed_point_repeats_geometry():
    zones = (BasicZone("a"), BasicZone("b"))
    polygon = Polygon([(0, 0), (100, 0), (100, 30), (70, 30), (70, 100), (0, 100)])
    arguments = _split_args(polygon, zones)
    arguments["zone_fixed_point"] = {zones[0]: Point(20, 20)}

    first_zones, first_roads = split_polygon(**arguments)
    second_zones, second_roads = split_polygon(**arguments)

    assert set(first_zones["zone"]) == set(zones)
    assert first_zones.geometry.is_valid.all()
    assert polygon.difference(first_zones.union_all()).area < 1e-6
    assert first_zones.geometry.normalize().to_wkb().tolist() == second_zones.geometry.normalize().to_wkb().tolist()
    assert first_roads.geometry.normalize().to_wkb().tolist() == second_roads.geometry.normalize().to_wkb().tolist()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"zone_ratios": {}}, "zone_ratios"),
        ({"zone_neighbors": [(BasicZone("a"), BasicZone("a"))]}, "self-pair"),
        ({"zone_fixed_point": {BasicZone("a"): Point()}}, "non-empty"),
    ],
)
def test_polygon_rejects_invalid_zone_configuration(changes, message):
    zones = (BasicZone("a"), BasicZone("b"))
    arguments = _split_args(box(0, 0, 100, 100), zones)
    arguments.update(changes)

    with pytest.raises(SplitPolygonValidationError, match=message):
        split_polygon(**arguments)
