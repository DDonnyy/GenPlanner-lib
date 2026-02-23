import json
import os
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.ops import voronoi_diagram
from shapely.strtree import STRtree
from tqdm.contrib.concurrent import process_map

matplotlib.use("Agg")

LOG_PATH = Path("../dev/test_mf2tz_101_f2tz_attempt_0.jsonl")

LOSS_YSCALE = "linear"  # "linear" / "log" / "symlog"
FPS = 60
DPI = 80
FRAME_STRIDE = 2

ZONE_COLORS = {
    "residential": "#FFD700",
    "industrial": "#6A5ACD",
    "business": "#FF8C00",
    "recreation": "#ADFF2F",
    "transport": "#A9A9A9",
    "agriculture": "#20B2AA",
    "special": "#8B4513",
}


def load_jsonl_log(path: Path) -> tuple[dict, pd.DataFrame]:
    """
    Returns:
      meta: dict from the first 'type=meta' record (or first record if it's meta)
      it_df: DataFrame with columns: iter(int), lr(float), loss_each_area..., and sites(list[dict])
    """
    meta = None
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            if obj.get("type") == "meta":
                meta = obj
                continue

            if "iter" in obj:
                rows.append(obj)
            else:
                continue

    if meta is None:
        raise ValueError(f"No meta record (type=meta) found in {path}")

    if not rows:
        raise ValueError(f"No iter records found in {path}")

    df = pd.DataFrame(rows)

    if "loss" in df.columns:
        loss_df = pd.json_normalize(df["loss"]).add_prefix("loss_")
        df = pd.concat([df.drop(columns=["loss"]), loss_df], axis=1)

    df["iter"] = df["iter"].astype(int)
    if "lr" in df.columns:
        df["lr"] = df["lr"].astype(float)

    return meta, df


def polygon_from_flat_xy(flat_xy: list[float]) -> Polygon:
    if len(flat_xy) < 6 or len(flat_xy) % 2 != 0:
        raise ValueError(f"polygon_coords must be even-length and >= 6 numbers, got {len(flat_xy)}")
    coords = [(float(flat_xy[i]), float(flat_xy[i + 1])) for i in range(0, len(flat_xy), 2)]
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.geom_type != "Polygon":
        raise ValueError(f"Boundary polygon invalid after fix. Type={poly.geom_type}, empty={poly.is_empty}")
    return poly


def load_loss_csv(loss_csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(loss_csv_path)
    if "iter" not in df.columns:
        raise ValueError(f"Loss CSV must contain 'iter' column. Got: {list(df.columns)}")
    df = df.copy()
    df["iter"] = df["iter"].astype(int)
    return df


def build_dense_tracks_from_jsonl(it_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      iters: (T,)
      sites: (N,) sorted site ids
      xy: (T, N, 2)
      zone_map: (N,) zone id per site (берём из первого появления)
    """
    it_df = it_df.sort_values("iter").reset_index(drop=True)

    # collect all site ids
    all_site_ids = set()
    for sites_list in it_df["sites"]:
        for s in sites_list:
            all_site_ids.add(int(s["i"]))
    sites = np.array(sorted(all_site_ids), dtype=np.int64)

    iters = it_df["iter"].to_numpy(dtype=np.int64)
    T = len(iters)
    N = len(sites)

    site2col = {int(s): j for j, s in enumerate(sites)}

    xy = np.full((T, N, 2), np.nan, dtype=np.float64)
    zone_map = np.full((N,), -1, dtype=np.int64)

    for t in range(T):
        for s in it_df.iloc[t]["sites"]:
            i = int(s["i"])
            j = site2col[i]
            xy[t, j, 0] = float(s["x"])
            xy[t, j, 1] = float(s["y"])
            if zone_map[j] < 0:
                zone_map[j] = int(s["z"])

    for j in range(N):
        for d in range(2):
            col = xy[:, j, d]
            for t in range(1, T):
                if np.isnan(col[t]) and np.isfinite(col[t - 1]):
                    col[t] = col[t - 1]
            for t in range(T - 2, -1, -1):
                if np.isnan(col[t]) and np.isfinite(col[t + 1]):
                    col[t] = col[t + 1]
            xy[:, j, d] = col

    if np.isnan(xy).any():
        bad = np.where(np.isnan(xy))
        raise ValueError(f"Still have NaNs in tracks; first bad index: {tuple(bad[i][0] for i in range(3))}")

    if (zone_map < 0).any():
        raise ValueError("Some sites never had zone id in logs (zone_map contains -1)")

    return iters, sites, xy, zone_map


def compute_voronoi_cells(points: list[Point], boundary: Polygon) -> list[Polygon]:
    mp = MultiPoint(points)
    diag = voronoi_diagram(mp, envelope=boundary, tolerance=0.0, edges=False)

    if diag.geom_type == "Polygon":
        polys = [diag]
    elif diag.geom_type == "MultiPolygon":
        polys = list(diag.geoms)
    elif diag.geom_type == "GeometryCollection":
        polys = [g for g in diag.geoms if g.geom_type == "Polygon"]
    else:
        raise ValueError(f"Unexpected voronoi_diagram output type: {diag.geom_type}")

    return polys


def assign_polygons_to_points(polys: list[Polygon], points: list[Point]) -> dict[int, Polygon]:
    tree = STRtree(polys)
    assignment: dict[int, Polygon] = {}

    for i, pt in enumerate(points):
        idxs = tree.query(pt)
        chosen = None

        for idx in idxs:
            poly = polys[int(idx)]
            if poly.contains(pt) or poly.covers(pt):
                chosen = poly
                break

        if chosen is None:
            best = None
            best_d = float("inf")
            for idx in idxs:
                poly = polys[int(idx)]
                d = poly.distance(pt)
                if d < best_d:
                    best_d = d
                    best = poly
            chosen = best

        if chosen is not None:
            assignment[i] = chosen

    if len(assignment) != len(points):
        missing = [i for i in range(len(points)) if i not in assignment]
        raise RuntimeError(f"Failed to assign polygons to points for indices: {missing[:20]} (total {len(missing)})")

    return assignment


def _iter_polygons(g):
    if g is None or g.is_empty:
        return
    gt = g.geom_type
    if gt == "Polygon":
        yield g
    elif gt == "MultiPolygon":
        for p in g.geoms:
            if not p.is_empty:
                yield p
    elif gt == "GeometryCollection":
        for gg in g.geoms:
            if gg.geom_type in ("Polygon", "MultiPolygon") and (not gg.is_empty):
                yield from _iter_polygons(gg)


def render_frame(
    boundary: Polygon,
    points_xy: np.ndarray,  # (N, 2)
    zone_per_site: np.ndarray,  # (N,)
    zone_id_to_name: dict[int, str],
    zone_colors: dict[str, str],
    out_path: str,
    dpi: int,
    xlim=None,
    ylim=None,
    loss_df: pd.DataFrame | None = None,
    loss_cols: list[str] | None = None,
    loss_yscale: str = "symlog",
    cur_iter: int | None = None,
    title_text: str | None = None,
):
    points = [Point(float(x), float(y)) for x, y in points_xy]
    polys = compute_voronoi_cells(points, boundary)
    mapping = assign_polygons_to_points(polys, points)

    fig, (ax, ax_loss) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(8, 10),
        dpi=dpi,
        gridspec_kw={"height_ratios": [3, 2]},
    )

    ax.set_aspect("equal", adjustable="box")

    for i, pt in enumerate(points):
        poly = mapping[i]
        zone_id = int(zone_per_site[i])
        zone_name = zone_id_to_name.get(zone_id, f"zone_{zone_id}")
        color = zone_colors.get(zone_name)

        clipped = poly.intersection(boundary)
        for p in _iter_polygons(clipped):
            xs, ys = p.exterior.xy
            ax.fill(xs, ys, alpha=0.75, linewidth=0.6, color=color, zorder=1)
            ax.plot(xs, ys, linewidth=0.6, color="black", zorder=2)

            for hole in p.interiors:
                hx, hy = hole.xy
                ax.fill(hx, hy, color="white", linewidth=0, zorder=1)

    bx, by = boundary.exterior.xy
    ax.plot(bx, by, linewidth=2.0, color="black")

    point_colors = []
    for i in range(len(points)):
        zone_id = int(zone_per_site[i])
        zone_name = zone_id_to_name.get(zone_id, f"zone_{zone_id}")
        point_colors.append(zone_colors.get(zone_name, "#CCCCCC"))

    ax.scatter(
        points_xy[:, 0],
        points_xy[:, 1],
        s=30,
        c=point_colors,
        edgecolors="black",
        linewidths=0.6,
        zorder=5,
    )

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.axis("off")

    if loss_df is not None and cur_iter is not None:
        if loss_cols is None:
            loss_cols = [c for c in loss_df.columns if c.startswith("loss_")]

        x_all = loss_df["iter"].to_numpy()

        all_losses = []

        for c in loss_cols:
            y_all = loss_df[c].to_numpy()
            ax_loss.plot(x_all, y_all, linewidth=1.0, alpha=0.6, label=c)
            y_all = y_all[np.isfinite(y_all)]
            if y_all.size > 0:
                all_losses.append(y_all)

        if all_losses:
            all_losses = np.concatenate(all_losses)
            y_top = float(np.percentile(all_losses, 98))
        else:
            y_top = 1.0

        if "lr" in loss_df.columns:
            lr = loss_df["lr"].to_numpy().astype(float)

            lr_min = float(np.nanmin(lr))
            lr_max = float(np.nanmax(lr))

            if np.isfinite(lr_min) and np.isfinite(lr_max) and lr_max > lr_min:
                lr_norm = (lr - lr_min) / (lr_max - lr_min)  # 0..1
                lr_scaled = lr_norm * y_top  # 0..y_top
                ax_loss.plot(x_all, lr_scaled, linewidth=1.2, alpha=0.35, linestyle="--", label="lr (scaled)")
            else:
                ax_loss.plot(
                    [x_all.min(), x_all.max()],
                    [0.05 * y_top, 0.05 * y_top],
                    linewidth=1.2,
                    alpha=0.35,
                    linestyle="--",
                    label="lr (const)",
                )

        ax_loss.axvline(cur_iter, linewidth=1.0)

        row = loss_df[loss_df["iter"] == cur_iter]
        if not row.empty:
            vals = []
            for c in loss_cols:
                v = float(row.iloc[0][c])
                ax_loss.scatter([cur_iter], [v], s=18)
                vals.append(f"{c}={v:.4g}")
            if "lr" in row.columns:
                vals.append(f"lr={float(row.iloc[0]['lr']):.4g}")

            ax_loss.text(
                0.99,
                0.95,
                "\n".join(vals),
                transform=ax_loss.transAxes,
                ha="right",
                va="top",
                multialignment="left",
                fontsize=6,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8),
            )

        ax_loss.set_xlim(x_all.min(), x_all.max())
        ax_loss.set_ylim(0, y_top if y_top > 0 else 1.0)
        if loss_yscale:
            ax_loss.set_yscale(loss_yscale)
        ax_loss.set_ylim(bottom=0)
        ax_loss.grid(True, alpha=0.3)
        ax_loss.set_xlabel("iter")
        ax_loss.set_ylabel("loss")
        ax_loss.legend(loc="upper left", fontsize=7, ncol=3)
    else:
        ax_loss.axis("off")

    if title_text:
        ax.text(
            0.01,
            0.99,
            title_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85),
            zorder=10,
        )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _render_one_frame(args):
    (
        k,
        it,
        points_xy,
        out_png,
        boundary,
        zone_map,
        zone_id_to_name,
        zone_colors,
        dpi,
        xlim,
        ylim,
        loss_df,
        loss_cols,
        loss_yscale,
        title_text,
    ) = args

    render_frame(
        boundary=boundary,
        points_xy=points_xy,
        zone_per_site=zone_map,
        zone_id_to_name=zone_id_to_name,
        zone_colors=zone_colors,
        out_path=out_png,
        dpi=dpi,
        xlim=xlim,
        ylim=ylim,
        loss_df=loss_df,
        loss_cols=loss_cols,
        loss_yscale=loss_yscale,
        cur_iter=int(it),
        title_text=title_text,
    )
    return out_png


def make_gif(
    log_path: Path,
    fps: int = 60,
    dpi: int = 80,
    loss_yscale: str = "linear",
    frame_stride: int = 1,
):
    if frame_stride < 1:
        raise ValueError("frame_stride must be >= 1")

    log_path = Path(log_path)
    run_name = log_path.stem
    out_gif = str(log_path.with_suffix(".gif"))

    meta, it_df = load_jsonl_log(log_path)

    if "polygon_coords" not in meta:
        raise ValueError("meta must contain polygon_coords")

    boundary = polygon_from_flat_xy(meta["polygon_coords"])
    iters, sites, xy, zone_map = build_dense_tracks_from_jsonl(it_df)

    loss_cols = [c for c in it_df.columns if c.startswith("loss_")]

    minx, miny, maxx, maxy = boundary.bounds
    padx = (maxx - minx) * 0.05 if maxx > minx else 1.0
    pady = (maxy - miny) * 0.05 if maxy > miny else 1.0
    global_xlim = (minx - padx, maxx + padx)
    global_ylim = (miny - pady, maxy + pady)

    title_text = f"{run_name}\nnormalize_rotation={meta.get('normalize_rotation')}, seed={meta.get('seed')}"

    os.makedirs(os.path.dirname(out_gif) or ".", exist_ok=True)
    tmp_dir = os.path.join(os.path.dirname(out_gif) or ".", f"_frames_{run_name}")
    os.makedirs(tmp_dir, exist_ok=True)

    iter_list = list(iters)

    iter_list = iter_list[::frame_stride]

    iter_to_row = {int(it): i for i, it in enumerate(iters)}
    zone_id_to_name = meta.get("zone_id_to_name")
    zone_id_to_name = {int(k): v for k, v in zone_id_to_name.items()}

    tasks = []
    for k, it in enumerate(iter_list):
        row = iter_to_row[int(it)]
        points_xy = xy[row]
        out_png = os.path.join(tmp_dir, f"frame_{k:05d}.png")

        tasks.append(
            (
                k,
                int(it),
                points_xy,
                out_png,
                boundary,
                zone_map,
                zone_id_to_name,
                ZONE_COLORS,
                dpi,
                global_xlim,
                global_ylim,
                it_df,
                loss_cols,
                loss_yscale,
                title_text,
            )
        )

    frame_paths = process_map(_render_one_frame, tasks, max_workers=os.cpu_count(), chunksize=1)

    images = [imageio.imread(p) for p in frame_paths]
    duration = 1 / max(1, fps)
    imageio.mimsave(out_gif, images, duration=duration)

    print(f"Saved GIF: {out_gif}")


if __name__ == "__main__":
    make_gif(LOG_PATH, fps=FPS, dpi=DPI, loss_yscale=LOSS_YSCALE, frame_stride=FRAME_STRIDE)
