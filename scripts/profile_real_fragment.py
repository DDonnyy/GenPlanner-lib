"""Profile one contour from the ignored scenario 1552 SVG, when available.

The SVG contains screen coordinates and already prepared contours rather than
the original projected input. This local diagnostic measures subdivision
complexity without storing source data in the repository. Run
``python scripts/profile_real_fragment.py``.
"""

import argparse
import cProfile
import hashlib
import io
import pstats
import re
import time
from pathlib import Path
from xml.etree import ElementTree

from shapely.geometry import Polygon

from genplanner.tasks.polygon_splitter import split_polygon
from genplanner.zones import BasicZone

DEFAULT_SVG = Path(__file__).resolve().parents[1] / "reports" / "scenario_1552" / "1552_input.svg"
COORDINATE_PATTERN = re.compile(r"(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)")


def read_simple_polygons(svg_path, fill):
    """Read single-ring M/L/Z SVG paths; skip paths with holes or curves."""
    root = ElementTree.parse(svg_path).getroot()
    polygons = []
    for element in root.iter():
        if not element.tag.endswith("path") or element.attrib.get("fill") != fill:
            continue
        commands = element.attrib.get("d", "")
        if commands.count("M") != 1 or commands.count("Z") != 1 or set(re.findall(r"[A-Za-z]", commands)) - set("MLZ"):
            continue
        polygon = Polygon([(float(x), float(y)) for x, y in COORDINATE_PATTERN.findall(commands)])
        if polygon.is_valid and polygon.area > 0:
            polygons.append(polygon)
    return polygons


def profile(label, function):
    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.enable()
    result = function()
    profiler.disable()
    elapsed = time.perf_counter() - start
    output = io.StringIO()
    pstats.Stats(profiler, stream=output).sort_stats("cumulative").print_stats(12)
    print(f"{label}: {elapsed:.3f}s")
    print(output.getvalue())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--svg", type=Path, default=DEFAULT_SVG)
    parser.add_argument("--fragment", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()
    if not args.svg.is_file():
        parser.error(f"scenario SVG does not exist: {args.svg}")

    territory_parts = read_simple_polygons(args.svg, "#ecd5aa")
    if args.fragment >= len(territory_parts) or args.fragment < 0:
        parser.error(f"fragment must be between 0 and {len(territory_parts) - 1}")
    fragment = territory_parts[args.fragment]
    print(f"fragment_vertices={len(fragment.exterior.coords)} area={fragment.area:.1f}")

    zones = [BasicZone("a"), BasicZone("b"), BasicZone("c")]
    generated, roads = profile(
        "split_polygon",
        lambda: split_polygon(
            polygon_to_split=fragment,
            zone_ratios={zone: 1 / 3 for zone in zones},
            zone_neighbors=[],
            zone_forbidden=[],
            zone_fixed_point={},
            local_crs=3857,
            run_name="profile_real_fragment",
            geom_simplify_tol=0.01,
            seed=7,
            max_iterations=args.iterations,
        ),
    )
    print(f"generated_zones={len(generated)} roads={len(roads)}")
    digest = hashlib.sha256()
    for zone, geometry in zip(generated["zone"], generated.geometry):
        digest.update(zone.name.encode("utf-8"))
        digest.update(geometry.normalize().wkb)
    for geometry in roads.geometry:
        digest.update(geometry.normalize().wkb)
    print(f"result_sha256={digest.hexdigest()}")


if __name__ == "__main__":
    main()
