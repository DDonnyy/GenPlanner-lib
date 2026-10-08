"""Repeat a fixed native optimization case and report time and output digest.

Run with ``python scripts/benchmark_rust_optimizer.py`` after ``maturin develop
--release``. Compare the digest before and after changing the Rust optimizer.
"""

import argparse
import hashlib
import statistics
import struct
import time

import numpy as np
from shapely.geometry import box

from genplanner._rust import optimize_territory_zoning
from genplanner.tasks.polygon_splitter import _sample_points_from_global_pool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--sites", type=int, default=15)
    parser.add_argument("--zones", type=int, default=3)
    args = parser.parse_args()
    if args.sites < args.zones or args.zones < 1:
        parser.error("sites must be at least zones, and zones must be positive")

    sites = _sample_points_from_global_pool(args.sites, seed=7).flatten().round(8).tolist()
    point2zone = np.repeat(np.arange(args.zones), (args.sites + args.zones - 1) // args.zones)[: args.sites]
    np.random.default_rng(7).shuffle(point2zone)
    boundary = [float(value) for xy in box(0, 0, 1, 1).exterior.normalize().coords[::-1] for value in xy]

    def optimize():
        return optimize_territory_zoning(
            boundary_xy=boundary,
            generator_points_xy=sites,
            point2zone=point2zone.tolist(),
            point_fixed_mask=[0.0] * (args.sites * 2),
            zone_target_area=[1 / args.zones] * args.zones,
            zone_neighbors=[],
            zone_forbidden=[],
            write_logs=False,
            run_name="benchmark_rust_optimizer",
            max_iterations=args.iterations,
        )

    optimize()  # warmup
    durations = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        result = optimize()
        durations.append(time.perf_counter() - start)

    coordinates, roads = result
    values = coordinates + [value for road in roads for value in road]
    digest = hashlib.sha256(struct.pack(f"<{len(values)}f", *values)).hexdigest()
    print(f"iterations={args.iterations} repeats={args.repeats} sites={args.sites} zones={args.zones}")
    print(f"median={statistics.median(durations):.6f}s min={min(durations):.6f}s max={max(durations):.6f}s")
    print(f"coordinates={len(coordinates)} roads={len(roads)} sha256={digest}")


if __name__ == "__main__":
    main()
