import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from shapely.geometry import Point, Polygon, MultiPoint
from shapely.ops import voronoi_diagram
from shapely.strtree import STRtree

import matplotlib.pyplot as plt
import imageio.v2 as imageio


from tqdm.contrib.concurrent import process_map

import matplotlib
matplotlib.use("Agg")


BOUNDARY_XY_FLAT = [
    0.0,
    0.44495219,
    0.01798835,
    0.37901258,
    0.03597671,
    0.31307297,
    0.08087802,
    0.25117152,
    0.12577933,
    0.18927008,
    0.19284429,
    0.20922147,
    0.25990925,
    0.22917287,
    0.31960231,
    0.17954815,
    0.41536495,
    0.20478421,
    0.51112759,
    0.23002027,
    0.60689024,
    0.25525632,
    0.70265288,
    0.28049238,
    0.79841552,
    0.30572844,
    0.89417817,
    0.3309645,
    0.98994081,
    0.35620056,
    0.98671038,
    0.43223306,
    0.98347995,
    0.50826557,
    0.98024952,
    0.58429807,
    1.0,
    0.63255385,
    0.99405254,
    0.72650285,
    0.98810508,
    0.82045185,
    0.88937156,
    0.81518912,
    0.79063805,
    0.80992639,
    0.69190454,
    0.80466366,
    0.59317103,
    0.79940093,
    0.49443751,
    0.79413819,
    0.40997919,
    0.79940093,
    0.32552087,
    0.80466366,
    0.24106254,
    0.80992639,
    0.15660422,
    0.81518912,
    0.0721459,
    0.82045185,
    0.0459602,
    0.72930028,
    0.0197745,
    0.63814871,
    0.00988725,
    0.54155045,
    0.0,
    0.44495219,
]

RUN_NAME = "test_run"

CSV_PATH = Path(f"../rust/{RUN_NAME}_sites.csv")
OUT_GIF = "./run.gif"

LOSS_CSV_PATH = Path(f"../rust/{RUN_NAME}.csv")
LOSS_COLUMNS = None
LOSS_YSCALE = "symlog"  # "linear" / "log" / "symlog"

FPS = 30
DPI = 100

ZONE_ID_TO_NAME = {
    0: "residential",
    1: "industrial",
    2: "business",
    3: "recreation",
    4: "transport",
    5: "agriculture",
    6: "special",
}


ZONE_COLORS = {
    "residential": "#FFD700",
    "industrial": "#6A5ACD",
    "business": "#FF8C00",
    "recreation": "#ADFF2F",
    "transport": "#A9A9A9",
    "agriculture": "#20B2AA",
    "special": "#8B4513",
}


@dataclass
class Config:
    boundary: list[float]
    csv_path: Path
    out_gif: str
    fps: int
    dpi: int
    zone_id_to_name: dict[int, str]
    zone_colors: dict[str, str]


def polygon_from_flat_xy(flat_xy: list[float]) -> Polygon:
    if len(flat_xy) < 6 or len(flat_xy) % 2 != 0:
        raise ValueError(f"BOUNDARY_XY_FLAT must be even-length and >= 6 numbers, got {len(flat_xy)}")

    coords = [(float(flat_xy[i]), float(flat_xy[i + 1])) for i in range(0, len(flat_xy), 2)]

    # на всякий случай замкнём контур
    if coords[0] != coords[-1]:
        coords.append(coords[0])

    poly = Polygon(coords)
    if not poly.is_valid:
        # лёгкая починка самопересечений/микроошибок
        poly = poly.buffer(0)

    if poly.is_empty or poly.geom_type != "Polygon":
        raise ValueError(f"Boundary polygon invalid after fix. Type={poly.geom_type}, empty={poly.is_empty}")

    return poly


def load_sites_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"iter", "site", "zone", "x", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}. Got columns: {list(df.columns)}")
    df = df.copy()
    df["iter"] = df["iter"].astype(int)
    df["site"] = df["site"].astype(int)
    df["zone"] = df["zone"].astype(int)
    df["x"] = df["x"].astype(float)
    df["y"] = df["y"].astype(float)
    return df


def load_loss_csv(loss_csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(loss_csv_path)
    if "iter" not in df.columns:
        raise ValueError(f"Loss CSV must contain 'iter' column. Got: {list(df.columns)}")
    df = df.copy()
    df["iter"] = df["iter"].astype(int)
    return df


def build_dense_tracks(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    iters: (T,)
    sites: (N,)
    xy: (T, N, 2) — координаты всех сайтов на каждой итерации (ffill/bfill)
    zone: (N,) — принадлежность сайта к зоне (берём first по site)
    """
    iters = np.sort(df["iter"].unique())
    sites = np.sort(df["site"].unique())

    pivot_x = (
        df.pivot_table(index="iter", columns="site", values="x", aggfunc="last").reindex(iters).reindex(columns=sites)
    )
    pivot_y = (
        df.pivot_table(index="iter", columns="site", values="y", aggfunc="last").reindex(iters).reindex(columns=sites)
    )

    pivot_x = pivot_x.ffill(axis=0).bfill(axis=0)
    pivot_y = pivot_y.ffill(axis=0).bfill(axis=0)

    if pivot_x.isna().any().any() or pivot_y.isna().any().any():
        nan_sites = sorted(
            set(pivot_x.columns[pivot_x.isna().any()].tolist() + pivot_y.columns[pivot_y.isna().any()].tolist())
        )
        raise ValueError(
            "Some sites never appear in the CSV, cannot reconstruct their positions. " f"Missing sites: {nan_sites}"
        )

    xy = np.stack([pivot_x.to_numpy(), pivot_y.to_numpy()], axis=-1)  # (T, N, 2)

    zone_map = df.sort_values(["site", "iter"]).groupby("site")["zone"].first().reindex(sites).to_numpy()

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


def render_frame(
    boundary: Polygon,
    points_xy: np.ndarray,  # (N, 2)
    zone_per_site: np.ndarray,  # (N,)
    zone_id_to_name: dict[int, str],
    zone_colors: dict[str, str],
    out_path: str,
    title: str,
    dpi: int,
    xlim=None,
    ylim=None,
    loss_df: pd.DataFrame | None = None,
    loss_cols: list[str] | None = None,
    loss_yscale: str = "symlog",
    cur_iter: int | None = None,
):
    points = [Point(float(x), float(y)) for x, y in points_xy]
    polys = compute_voronoi_cells(points, boundary)
    mapping = assign_polygons_to_points(polys, points)

    fig, (ax, ax_loss) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(8, 10),
        dpi=dpi,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)

    # 1 полигон на 1 сайт
    for i, pt in enumerate(points):
        poly = mapping[i]
        zone_id = int(zone_per_site[i])
        zone_name = zone_id_to_name.get(zone_id, f"zone_{zone_id}")
        color = zone_colors.get(zone_name)

        xs, ys = poly.exterior.xy
        ax.fill(xs, ys, alpha=0.75, linewidth=0.6, color=color)
        ax.plot(xs, ys, linewidth=0.6, color="black")

    # boundary
    bx, by = boundary.exterior.xy
    ax.plot(bx, by, linewidth=2.0, color="black")

    ax.scatter(points_xy[:, 0], points_xy[:, 1], s=10, color="black")

    # minx, miny, maxx, maxy = boundary.bounds
    # padx = (maxx - minx) * 0.02 if maxx > minx else 1.0
    # pady = (maxy - miny) * 0.02 if maxy > miny else 1.0
    # ax.set_xlim(minx - padx, maxx + padx)
    # ax.set_ylim(miny - pady, maxy + pady)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.axis("off")

    # ----- LOSS PLOT (bottom panel) -----
    if loss_df is not None and cur_iter is not None:
        if loss_cols is None:
            loss_cols = [c for c in loss_df.columns if c.startswith("loss_")]

        # рисуем весь график (фон)
        x_all = loss_df["iter"].to_numpy()
        for c in loss_cols:
            y_all = loss_df[c].to_numpy()
            ax_loss.plot(x_all, y_all, linewidth=1.0, alpha=0.6, label=c)

        # выделяем текущую итерацию
        ax_loss.axvline(cur_iter, linewidth=1.0)

        row = loss_df[loss_df["iter"] == cur_iter]
        if not row.empty:
            vals = []
            for c in loss_cols:
                v = float(row.iloc[0][c])
                ax_loss.scatter([cur_iter], [v], s=18)
                vals.append(f"{c}={v:.4g}")

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
        if loss_yscale:
            ax_loss.set_yscale(loss_yscale)
        ax_loss.set_ylim(bottom=0)
        ax_loss.grid(True, alpha=0.3)
        ax_loss.set_xlabel("iter")
        ax_loss.set_ylabel("loss")
        ax_loss.legend(loc="upper left", fontsize=7, ncol=3)
    else:
        ax_loss.axis("off")

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
    ) = args

    render_frame(
        boundary=boundary,
        points_xy=points_xy,
        zone_per_site=zone_map,
        zone_id_to_name=zone_id_to_name,
        zone_colors=zone_colors,
        out_path=out_png,
        title=f"iter {int(it)}",
        dpi=dpi,
        xlim=xlim,
        ylim=ylim,
        loss_df=loss_df,
        loss_cols=loss_cols,
        loss_yscale=loss_yscale,
        cur_iter=int(it),
    )
    return out_png


def make_gif(cfg: Config):
    boundary = polygon_from_flat_xy(cfg.boundary)

    df = load_sites_csv(cfg.csv_path)
    iters, sites, xy, zone_map = build_dense_tracks(df)

    loss_df = load_loss_csv(LOSS_CSV_PATH)
    loss_cols = LOSS_COLUMNS
    if loss_cols is None:
        loss_cols = [c for c in loss_df.columns if c.startswith("loss_")]

    os.makedirs(os.path.dirname(cfg.out_gif) or ".", exist_ok=True)
    tmp_dir = os.path.join(os.path.dirname(cfg.out_gif) or ".", "_frames_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    iter_list = list(iters)

    iter_to_row = {int(it): i for i, it in enumerate(iters)}

    # глобальные границы по всем точкам за все итерации
    all_x = xy[:, :, 0]
    all_y = xy[:, :, 1]

    minx = float(all_x.min())
    maxx = float(all_x.max())
    miny = float(all_y.min())
    maxy = float(all_y.max())

    padx = (maxx - minx) * 0.05 if maxx > minx else 1.0
    pady = (maxy - miny) * 0.05 if maxy > miny else 1.0

    global_xlim = (minx - padx, maxx + padx)
    global_ylim = (miny - pady, maxy + pady)

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
                cfg.zone_id_to_name,
                cfg.zone_colors,
                cfg.dpi,
                global_xlim,
                global_ylim,
                loss_df,
                loss_cols,
                LOSS_YSCALE,
            )
        )

    frame_paths = process_map(
        _render_one_frame,
        tasks,
        max_workers=os.cpu_count(),
    )

    images = [imageio.imread(p) for p in frame_paths]
    duration = 1.0 / max(1, cfg.fps)
    imageio.mimsave(cfg.out_gif, images, duration=duration)


if __name__ == "__main__":
    cfg = Config(
        boundary=BOUNDARY_XY_FLAT,
        csv_path=CSV_PATH,
        out_gif=OUT_GIF,
        fps=FPS,
        dpi=DPI,
        zone_id_to_name=ZONE_ID_TO_NAME,
        zone_colors=ZONE_COLORS,
    )
    make_gif(cfg)
    print(f"Saved GIF: {OUT_GIF}")
