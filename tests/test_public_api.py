"""Small offline checks for the public Python and Rust-backed package."""

import tomllib
from importlib.metadata import version
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

from genplanner import GenPlanner, Relation, ZoneRelationMatrix
from genplanner.zones import BasicZone


def test_package_versions_agree():
    root = Path(__file__).resolve().parents[1]
    with (root / "pyproject.toml").open("rb") as project_file:
        project_version = tomllib.load(project_file)["project"]["version"]
    with (root / "rust" / "Cargo.toml").open("rb") as cargo_file:
        cargo_version = tomllib.load(cargo_file)["package"]["version"]

    assert project_version == cargo_version == version("genplanner")


def test_relation_matrix_preserves_symmetric_forbidden_pair():
    residential = BasicZone("residential")
    industrial = BasicZone("industrial")

    matrix = ZoneRelationMatrix.from_pairs(zones=[residential, industrial], forbidden=[(residential, industrial)])

    assert matrix.get(residential, industrial) == Relation.FORBIDDEN
    assert matrix.get(industrial, residential) == Relation.FORBIDDEN


def test_planner_prepares_polygon_in_projected_crs():
    features = gpd.GeoDataFrame(geometry=[box(30, 59, 30.01, 59.01)], crs="EPSG:4326")

    planner = GenPlanner(features_gdf=features, parallel=False)

    assert len(planner.territory_to_work_with) == 1
    assert planner.territory_to_work_with.crs.is_projected
    assert planner.original_crs == features.crs
