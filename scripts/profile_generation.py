"""Profile representative geometry work without external input data.

Run with ``python scripts/profile_generation.py`` from an installed development environment.
"""

import cProfile
import io
import pstats
import queue
import time

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, box

from genplanner.main.genplanner import split_queue
from genplanner.tasks.polygon_splitter import split_polygon
from genplanner.utils import elastic_wrap, territory_splitter
from genplanner.zones import BasicZone


def profile(label, function):
    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.enable()
    result = function()
    profiler.disable()
    elapsed = time.perf_counter() - start
    output = io.StringIO()
    pstats.Stats(profiler, stream=output).sort_stats("cumulative").print_stats(15)
    print(f"{label}: {elapsed:.3f}s")
    print(output.getvalue())
    return result


def empty_task(task, **kwargs):
    return {}


def main():
    pending = queue.Queue()
    for _ in range(300):
        pending.put((empty_task, (), {}))
    profile("serial queue: 300 empty tasks", lambda: split_queue(pending, 3857, parallel=False, max_workers=1))

    separated = gpd.GeoDataFrame(geometry=[box(i * 110, 0, i * 110 + 100, 100) for i in range(1000)], crs=3857)
    profile("elastic_wrap: 1000 components", lambda: elastic_wrap(separated))

    road_count = 5000
    road_lines = gpd.GeoDataFrame(
        {"roads_width": np.arange(road_count) % 10 + 1},
        geometry=[LineString([(i, 0), (i, 100)]) for i in range(road_count)],
        crs=3857,
    )
    profile(
        "road buffers: 5000 lines",
        lambda: road_lines.geometry.buffer(road_lines.roads_width / 2, quad_segs=4),
    )

    territory = gpd.GeoDataFrame(
        {"zone": list(range(20))},
        geometry=[box(i * 100, 0, (i + 1) * 100, 100) for i in range(20)],
        crs=3857,
    )
    roads = gpd.GeoDataFrame(
        geometry=[LineString([(-10, y), (2010, y)]).buffer(2) for y in range(10, 100, 10)], crs=3857
    )
    profile(
        "territory_splitter", lambda: territory_splitter(territory, roads, reproject_attr=True, select_by_point=True)
    )

    zones = [BasicZone("a"), BasicZone("b"), BasicZone("c")]
    profile(
        "split_polygon",
        lambda: split_polygon(
            polygon_to_split=box(0, 0, 1000, 1000),
            zone_ratios={zone: 1 / 3 for zone in zones},
            zone_neighbors=[],
            zone_forbidden=[],
            zone_fixed_point={},
            local_crs=3857,
            run_name="profile_generation",
            geom_simplify_tol=0.01,
            seed=7,
            max_iterations=200,
        ),
    )


if __name__ == "__main__":
    main()
