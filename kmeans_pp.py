"""Constrained K-Means++ for new supply location-allocation.

Space modes
-----------
euclidean  Straight-line distance in the input x/y coordinate space.
geometric  Input must be lon/lat; coordinates are projected to the local UTM
           zone and straight-line distances are in metres.
network    Input must be lon/lat; a street network is downloaded from
           OpenStreetMap (via OSMnx), every point is edge-snapped to the
           nearest street segment, and distances are shortest-path travel
           lengths in metres.  New supply locations are placed on the network
           at the snapped edge position of the optimal demand medoid.

Edge snapping (network mode)
----------------------------
Each point is orthogonally projected onto its nearest OSM edge (the
perpendicular foot-point).  The snapped position retains full network
connectivity: distances to both edge endpoints are stored and used when
computing shortest-path distances through the graph.

Problem
-------
You have demand points and existing supply points (frozen cluster centres).
You want to add N new supply points so that demand is served as efficiently
as possible, without moving the existing centres.

Algorithm
---------
1. K-Means++ seeding for the N new centres (D(x) is the distance from a
   demand point to the nearest already-placed centre; sampling probability
   ∝ weight(x) · D(x)²).
2. Constrained Lloyd's iteration: assign each demand to its nearest centre,
   then update ONLY the new centres (weighted mean for Euclidean/geometric,
   weighted medoid for network).  Existing centres never move.

Dependencies
------------
All modes   numpy  pandas  matplotlib
geometric   pyproj
network     osmnx  networkx  shapely  pyproj

Usage
-----
python kmeans_pp.py --demand demand.csv --supply supply.csv --n-new 3
python kmeans_pp.py --demand demand.csv --supply supply.csv --n-new 3 --space geometric
python kmeans_pp.py --demand demand.csv --supply supply.csv --n-new 3 --space network
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Edit the values below, then run:  python kmeans_pp.py

DEMAND       = "demand.csv"          # path to demand CSV
SUPPLY       = "supply.csv"          # path to existing supply CSV
N_NEW        = 2                     # number of new supply centres to add

# Distance space: "euclidean" | "geometric" | "network"
#   euclidean  — plain x/y coordinates
#   geometric  — lon/lat input, metric distances via UTM projection
#   network    — lon/lat input, real-world street distances via OSM
SPACE        = "euclidean"

WEIGHT_COL   = "weight"             # column in demand CSV to use as weight
                                    # (set to "" to ignore weights / treat all as 1)

NETWORK_TYPE = "drive"              # OSM network type: drive | walk | bike | all
BUFFER       = 500                  # metres of padding around OSM bounding box

N_INIT       = 10                   # random restarts — best result is kept
MAX_ITER     = 300                  # max Lloyd iterations per restart
SEED         = 0                    # random seed for reproducibility

OUTPUT       = "demand_clustered.csv"       # output CSV path
FIGURE       = "clusters_before_after.png"  # output figure path

# ─────────────────────────────────────────────────────────────────────────────

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SpaceType = Literal["euclidean", "geometric", "network"]

# ── optional heavy dependencies ───────────────────────────────────────────────

try:
    import networkx as nx
    import osmnx as ox
    from shapely.geometry import LineString, Point
    _HAS_NET = True
    _OSMNX_MAJOR = int(ox.__version__.split(".")[0])
except ImportError:
    _HAS_NET = False
    _OSMNX_MAJOR = 0

try:
    from pyproj import Transformer as _Transformer
    _HAS_PROJ = True
except ImportError:
    _HAS_PROJ = False


# ── edge-snap data class ──────────────────────────────────────────────────────

@dataclass
class EdgeSnap:
    """One input point after snapping to the nearest OSM edge (projected graph)."""
    proj_x: float    # projected (UTM) x of the foot-point [m]
    proj_y: float    # projected (UTM) y of the foot-point [m]
    snap_lon: float  # back-projected longitude of the foot-point
    snap_lat: float  # back-projected latitude  of the foot-point
    u: int           # from-node osmid of the host edge
    v: int           # to-node  osmid of the host edge
    key: int         # multigraph edge key (usually 0)
    dist_u: float    # distance from foot-point to node u along the edge [m]
    dist_v: float    # distance from foot-point to node v along the edge [m]


# ── projection helpers ────────────────────────────────────────────────────────

def _make_utm_projectors(all_lons: np.ndarray, all_lats: np.ndarray):
    """Return (forward, inverse) pyproj Transformers for the local UTM zone."""
    if not _HAS_PROJ:
        raise ImportError(
            "pyproj is required for geometric and network space.\n"
            "  pip install pyproj"
        )
    clon = float(np.mean(all_lons))
    clat = float(np.mean(all_lats))
    zone = int((clon + 180) / 6) + 1
    hemi = "north" if clat >= 0 else "south"
    utm  = f"+proj=utm +zone={zone} +{hemi} +ellps=WGS84"
    fwd  = _Transformer.from_crs("EPSG:4326", utm, always_xy=True)
    inv  = _Transformer.from_crs(utm, "EPSG:4326", always_xy=True)
    return fwd, inv


# ── OSMnx / network helpers ───────────────────────────────────────────────────

def _fetch_graph(
    lons: np.ndarray,
    lats: np.ndarray,
    network_type: str = "drive",
    buffer_m: float = 500,
):
    """Download and project the OSM street network covering all given points.

    Returns (G, G_proj) where G is in WGS-84 (lon/lat) and G_proj is in the
    graph's local projected CRS (metres).
    """
    if not _HAS_NET:
        raise ImportError(
            "osmnx, networkx and shapely are required for network space.\n"
            "  pip install osmnx networkx shapely"
        )
    margin = max(1e-4, buffer_m / 111_000)   # degrees ≈ buffer_m metres
    north  = float(lats.max()) + margin
    south  = float(lats.min()) - margin
    east   = float(lons.max()) + margin
    west   = float(lons.min()) - margin

    print(
        f"Fetching OSM network  ({south:.4f}–{north:.4f} N, "
        f"{west:.4f}–{east:.4f} E)  type={network_type} …"
    )
    if _OSMNX_MAJOR >= 2:
        from shapely.geometry import box as _box
        G = ox.graph_from_polygon(_box(west, south, east, north),
                                   network_type=network_type)
    else:
        G = ox.graph_from_bbox(north, south, east, west,
                               network_type=network_type,
                               buffer_dist=buffer_m)

    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges — projecting …")
    G_proj = ox.project_graph(G)
    return G, G_proj


def _snap_to_edges(
    G_proj,
    proj_xs: np.ndarray,
    proj_ys: np.ndarray,
    back_proj,
) -> list[EdgeSnap]:
    """Snap each projected point to its nearest edge in G_proj.

    back_proj is a pyproj Transformer (projected CRS → EPSG:4326).
    """
    xs = np.atleast_1d(proj_xs).astype(float)
    ys = np.atleast_1d(proj_ys).astype(float)

    if _OSMNX_MAJOR >= 2:
        nearest = ox.nearest_edges(G_proj, xs, ys)
    else:
        nearest = ox.nearest_edges(G_proj, xs, ys, return_dist=False)

    snaps: list[EdgeSnap] = []
    for i, (u, v, key) in enumerate(nearest):
        ed = G_proj[u][v][key]
        if "geometry" in ed:
            line: LineString = ed["geometry"]
        else:
            ux, uy = G_proj.nodes[u]["x"], G_proj.nodes[u]["y"]
            vx, vy = G_proj.nodes[v]["x"], G_proj.nodes[v]["y"]
            line = LineString([(ux, uy), (vx, vy)])

        pt     = Point(xs[i], ys[i])
        d_u    = line.project(pt)          # distance along edge from u end
        foot   = line.interpolate(d_u)
        sx, sy = foot.x, foot.y

        if back_proj is not None:
            slon, slat = back_proj.transform(sx, sy)
        else:
            slon, slat = sx, sy

        snaps.append(EdgeSnap(
            proj_x=float(sx),  proj_y=float(sy),
            snap_lon=float(slon), snap_lat=float(slat),
            u=int(u), v=int(v), key=int(key),
            dist_u=float(d_u),
            dist_v=float(max(line.length - d_u, 0.0)),
        ))
    return snaps


def _precompute_node_dists(
    G_proj,
    snaps: list[EdgeSnap],
) -> dict[int, dict[int, float]]:
    """Dijkstra from every unique node that is an endpoint of a snapped edge."""
    nodes = {s.u for s in snaps} | {s.v for s in snaps}
    print(f"  Running Dijkstra from {len(nodes)} nodes (this may take a moment) …")
    return {
        n: dict(nx.single_source_dijkstra_path_length(G_proj, n, weight="length"))
        for n in nodes
    }


def _snap_dist(
    a: EdgeSnap,
    b: EdgeSnap,
    nd: dict[int, dict[int, float]],
) -> float:
    """Shortest network distance between two edge-snapped points [m]."""
    cands: list[float] = []

    # Same-edge shortcut: travel directly along the shared edge
    if a.u == b.u and a.v == b.v and a.key == b.key:
        cands.append(abs(a.dist_u - b.dist_u))

    # Routes via pairs of (a-endpoint, b-endpoint)
    for da, ua in [(a.dist_u, a.u), (a.dist_v, a.v)]:
        for db, ub in [(b.dist_u, b.u), (b.dist_v, b.v)]:
            via = nd.get(ua, {}).get(ub, np.inf)
            cands.append(da + via + db)

    return float(min(cands))


def _build_dist_matrix(
    snaps: list[EdgeSnap],
    nd: dict[int, dict[int, float]],
) -> np.ndarray:
    n = len(snaps)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _snap_dist(snaps[i], snaps[j], nd)
            D[i, j] = d
            D[j, i] = d
    return D


# ── Euclidean core (euclidean / geometric modes) ──────────────────────────────

def _eu_nearest_sq(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    diff = X[:, None, :] - centers[None, :, :]
    return np.min(np.sum(diff ** 2, axis=2), axis=1)


def _eu_assign(X: np.ndarray, centers: np.ndarray):
    diff = X[:, None, :] - centers[None, :, :]
    sq   = np.sum(diff ** 2, axis=2)
    lbl  = np.argmin(sq, axis=1)
    return lbl, sq[np.arange(len(X)), lbl]


def _eu_kpp_init(
    X: np.ndarray,
    weights: np.ndarray,
    fixed_centers: np.ndarray,
    n_new: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n, k = X.shape
    nc   = np.empty((n_new, k))
    sq   = _eu_nearest_sq(X, fixed_centers)
    for i in range(n_new):
        scores = weights * sq
        tot    = scores.sum()
        idx    = int(rng.integers(n) if tot <= 0 else rng.choice(n, p=scores / tot))
        nc[i]  = X[idx]
        sq     = np.minimum(sq, np.sum((X - nc[i]) ** 2, axis=1))
    return nc


def constrained_kmeans_euclidean(
    X: np.ndarray,
    weights: np.ndarray,
    fixed_centers: np.ndarray,
    n_new: int,
    max_iter: int = 300,
    tol: float = 1e-6,
    n_init: int = 10,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Constrained K-Means (Euclidean).  Returns (new_centers, labels, inertia)."""
    rng     = np.random.default_rng(random_state)
    n_fixed = len(fixed_centers)
    best: tuple[np.ndarray, np.ndarray, float] | None = None

    for _ in range(n_init):
        nc   = _eu_kpp_init(X, weights, fixed_centers, n_new, rng)
        prev = np.inf

        for _ in range(max_iter):
            all_c           = np.vstack([fixed_centers, nc])
            lbl, sq         = _eu_assign(X, all_c)
            inertia         = float((weights * sq).sum())

            updated = nc.copy()
            for j in range(n_new):
                mask = lbl == (n_fixed + j)
                if mask.any():
                    w          = weights[mask]
                    updated[j] = (X[mask] * w[:, None]).sum(0) / w.sum()
                else:
                    updated[j] = X[np.argmax(weights * _eu_nearest_sq(X, all_c))]

            shift = float(np.sum((updated - nc) ** 2))
            nc    = updated
            if abs(prev - inertia) < tol or shift < tol:
                break
            prev = inertia

        all_c   = np.vstack([fixed_centers, nc])
        lbl, sq = _eu_assign(X, all_c)
        inertia = float((weights * sq).sum())
        if best is None or inertia < best[2]:
            best = (nc.copy(), lbl.copy(), inertia)

    assert best is not None
    return best


# ── Matrix-based core (network mode) ─────────────────────────────────────────

def _mat_nearest(D: np.ndarray, cidxs: np.ndarray) -> np.ndarray:
    return D[:, cidxs].min(axis=1)


def _mat_assign(D: np.ndarray, cidxs: np.ndarray):
    sub  = D[:, cidxs]
    lbl  = np.argmin(sub, axis=1)
    dist = sub[np.arange(len(D)), lbl]
    return lbl, dist


def _mat_kpp_init(
    D: np.ndarray,
    weights: np.ndarray,
    fixed_idxs: np.ndarray,
    n_new: int,
    demand_idxs: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    md     = _mat_nearest(D, fixed_idxs)
    chosen: list[int] = []
    for _ in range(n_new):
        scores = weights * md ** 2
        tot    = scores.sum()
        idx    = (int(rng.choice(demand_idxs))
                  if tot <= 0
                  else int(rng.choice(len(D), p=scores / tot)))
        chosen.append(idx)
        md = np.minimum(md, D[:, idx])
    return np.array(chosen, dtype=int)


def _medoid(D: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> int:
    """Index of the weighted medoid among masked rows.

    Minimises sum_j  w_j · d(i, j)  over all members j in the cluster.
    """
    idxs = np.where(mask)[0]
    if len(idxs) == 1:
        return int(idxs[0])
    sub  = D[np.ix_(idxs, idxs)]
    # cost[row] = Σ_col  w[col] * d(row, col)
    cost = (sub * weights[idxs]).sum(axis=1)
    return int(idxs[np.argmin(cost)])


def constrained_kmeans_matrix(
    D: np.ndarray,
    weights: np.ndarray,
    n_fixed: int,
    n_new: int,
    max_iter: int = 300,
    tol: float = 1e-6,
    n_init: int = 10,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Constrained K-Means using a precomputed distance matrix.

    The first n_fixed rows/columns of D are existing supply (fixed centres).
    The remaining rows are demand points.  Returns (new_center_indices, labels,
    inertia) where indices point into the full D matrix.
    """
    rng         = np.random.default_rng(random_state)
    fixed_idxs  = np.arange(n_fixed, dtype=int)
    demand_idxs = np.arange(n_fixed, len(D), dtype=int)
    best: tuple[np.ndarray, np.ndarray, float] | None = None

    for _ in range(n_init):
        nc_idxs = _mat_kpp_init(D, weights, fixed_idxs, n_new, demand_idxs, rng)
        prev    = np.inf

        for _ in range(max_iter):
            all_c        = np.concatenate([fixed_idxs, nc_idxs])
            lbl, dists   = _mat_assign(D, all_c)
            inertia      = float((weights * dists).sum())

            updated = nc_idxs.copy()
            for j in range(n_new):
                label_j = n_fixed + j
                mask    = lbl == label_j
                if mask.any():
                    updated[j] = _medoid(D, weights, mask)
                else:
                    mind       = _mat_nearest(D, all_c)
                    updated[j] = int(demand_idxs[
                        np.argmax(weights[demand_idxs] * mind[demand_idxs])
                    ])

            shift   = int(np.sum(updated != nc_idxs))
            nc_idxs = updated
            if abs(prev - inertia) < tol or shift == 0:
                break
            prev = inertia

        all_c      = np.concatenate([fixed_idxs, nc_idxs])
        lbl, dists = _mat_assign(D, all_c)
        inertia    = float((weights * dists).sum())
        if best is None or inertia < best[2]:
            best = (nc_idxs.copy(), lbl.copy(), inertia)

    assert best is not None
    return best


# ── I/O ───────────────────────────────────────────────────────────────────────

def _load(
    path: Path,
    x_col: str = "x",
    y_col: str = "y",
    weight_col: str = "weight",
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None]:
    df = pd.read_csv(path)
    for c in (x_col, y_col):
        if c not in df.columns:
            raise ValueError(f"{path}: missing required column '{c}'")
    coords  = df[[x_col, y_col]].to_numpy(float)
    weights = df[weight_col].to_numpy(float) if weight_col in df.columns else None
    return df, coords, weights


# ── plotting ──────────────────────────────────────────────────────────────────

def _draw_network_edges(G, ax) -> None:
    """Draw OSM edges lightly as background on ax.

    G should be the *unprojected* (WGS-84) graph so its node coordinates
    are already in lon/lat, matching the scatter-plot axes.
    """
    nodes_xy = {n: (d["x"], d["y"]) for n, d in G.nodes(data=True)}
    for u, v, ed in G.edges(data=True):
        if "geometry" in ed:
            xs, ys = ed["geometry"].xy
        else:
            xs = [nodes_xy[u][0], nodes_xy[v][0]]
            ys = [nodes_xy[u][1], nodes_xy[v][1]]
        ax.plot(xs, ys, color="#cccccc", linewidth=0.5, zorder=1)


def _plot(
    demand_xy: np.ndarray,
    fixed_xy: np.ndarray,
    new_xy: np.ndarray,
    lbl_before: np.ndarray,
    lbl_after: np.ndarray,
    save: Path | None,
    space: SpaceType,
    G=None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

    if space == "network" and G is not None and _HAS_NET:
        for ax in axes:
            _draw_network_edges(G, ax)

    panels = [
        (axes[0], lbl_before, "Before — existing supply only",     False),
        (axes[1], lbl_after,  "After  — existing + new supply",    True),
    ]
    for ax, lbl, title, show_new in panels:
        ax.scatter(demand_xy[:, 0], demand_xy[:, 1],
                   c=lbl, s=18, cmap="tab20", alpha=0.8, zorder=3)
        ax.scatter(fixed_xy[:, 0], fixed_xy[:, 1],
                   marker="s", s=200, c="black", edgecolors="white", linewidths=1.5,
                   label="existing supply", zorder=5)
        if show_new:
            ax.scatter(new_xy[:, 0], new_xy[:, 1],
                       marker="*", s=320, c="red", edgecolors="white", linewidths=1.5,
                       label="new supply", zorder=5)
        ax.set_title(title)
        ax.legend(loc="best")

    xlabel, ylabel = (
        ("lon", "lat") if space in ("geometric", "network") else ("x", "y")
    )
    for ax in axes:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
        print(f"Saved figure → {save}")
    plt.show()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    demand_path  = Path(DEMAND)
    supply_path  = Path(SUPPLY)
    n_new        = int(N_NEW)
    space: SpaceType = SPACE          # type: ignore[assignment]
    weight_col   = WEIGHT_COL or None
    network_type = NETWORK_TYPE
    buffer       = float(BUFFER)
    n_init       = int(N_INIT)
    max_iter     = int(MAX_ITER)
    seed         = int(SEED)
    output       = Path(OUTPUT)
    figure       = Path(FIGURE)

    if n_new < 1:
        raise SystemExit("N_NEW must be >= 1")
    if space not in ("euclidean", "geometric", "network"):
        raise SystemExit(f"SPACE must be 'euclidean', 'geometric', or 'network', got {space!r}")

    demand_df, demand_coords, demand_weights = _load(
        demand_path, weight_col=weight_col or "weight"
    )
    if demand_weights is None or weight_col == "":
        demand_weights = np.ones(len(demand_coords))
    _, supply_coords, _ = _load(supply_path)
    n_supply = len(supply_coords)

    # ── run algorithm ─────────────────────────────────────────────────────────

    G_plot     = None   # unprojected graph kept for plotting (network mode)
    snaps_demand: list[EdgeSnap] | None = None

    if space == "euclidean":
        X, FC = demand_coords, supply_coords

        lbl_before, _             = _eu_assign(X, FC)
        new_c, lbl_after, inertia = constrained_kmeans_euclidean(
            X, demand_weights, FC, n_new,
            max_iter=max_iter, n_init=n_init, random_state=seed,
        )
        demand_xy, fixed_xy, new_xy = X, FC, new_c

    elif space == "geometric":
        all_lons = np.concatenate([demand_coords[:, 0], supply_coords[:, 0]])
        all_lats = np.concatenate([demand_coords[:, 1], supply_coords[:, 1]])
        fwd, inv = _make_utm_projectors(all_lons, all_lats)

        dx, dy = fwd.transform(demand_coords[:, 0], demand_coords[:, 1])
        sx, sy = fwd.transform(supply_coords[:, 0], supply_coords[:, 1])
        X, FC  = np.column_stack([dx, dy]), np.column_stack([sx, sy])

        lbl_before, _                  = _eu_assign(X, FC)
        new_c_proj, lbl_after, inertia = constrained_kmeans_euclidean(
            X, demand_weights, FC, n_new,
            max_iter=max_iter, n_init=n_init, random_state=seed,
        )
        nc_lons, nc_lats = inv.transform(new_c_proj[:, 0], new_c_proj[:, 1])
        new_c            = np.column_stack([nc_lons, nc_lats])

        demand_xy = demand_coords
        fixed_xy  = supply_coords
        new_xy    = new_c

    elif space == "network":
        all_lons = np.concatenate([demand_coords[:, 0], supply_coords[:, 0]])
        all_lats = np.concatenate([demand_coords[:, 1], supply_coords[:, 1]])

        G_plot, G_proj = _fetch_graph(
            all_lons, all_lats,
            network_type=network_type,
            buffer_m=buffer,
        )

        graph_crs = G_proj.graph.get("crs")
        if graph_crs and _HAS_PROJ:
            fwd_proj  = _Transformer.from_crs("EPSG:4326", graph_crs, always_xy=True)
            back_proj = _Transformer.from_crs(graph_crs, "EPSG:4326", always_xy=True)
            dx, dy    = fwd_proj.transform(demand_coords[:, 0], demand_coords[:, 1])
            sx, sy    = fwd_proj.transform(supply_coords[:, 0], supply_coords[:, 1])
        else:
            fwd_utm, back_proj = _make_utm_projectors(all_lons, all_lats)
            dx, dy = fwd_utm.transform(demand_coords[:, 0], demand_coords[:, 1])
            sx, sy = fwd_utm.transform(supply_coords[:, 0], supply_coords[:, 1])

        all_px = np.concatenate([sx, dx])
        all_py = np.concatenate([sy, dy])
        print(f"Snapping {len(all_px)} points to nearest edge …")
        all_snaps    = _snap_to_edges(G_proj, all_px, all_py, back_proj)
        snaps_supply = all_snaps[:n_supply]
        snaps_demand = all_snaps[n_supply:]

        nd = _precompute_node_dists(G_proj, all_snaps)
        print(f"Building {len(all_snaps)}×{len(all_snaps)} network distance matrix …")
        D = _build_dist_matrix(all_snaps, nd)

        w_full = np.concatenate([np.zeros(n_supply), demand_weights])

        nc_idxs, lbl_all, inertia = constrained_kmeans_matrix(
            D, w_full, n_fixed=n_supply, n_new=n_new,
            max_iter=max_iter, n_init=n_init, random_state=seed,
        )

        lbl_after  = lbl_all[n_supply:]
        lbl_before = np.argmin(D[n_supply:, :n_supply], axis=1)

        new_snaps = [all_snaps[i] for i in nc_idxs]
        new_c     = np.array([[s.snap_lon, s.snap_lat] for s in new_snaps])
        demand_xy = np.array([[s.snap_lon, s.snap_lat] for s in snaps_demand])
        fixed_xy  = np.array([[s.snap_lon, s.snap_lat] for s in snaps_supply])
        new_xy    = new_c

    else:
        raise SystemExit(f"Unknown SPACE: {space!r}")

    # ── report ────────────────────────────────────────────────────────────────

    print(f"\nFinal inertia (weighted distance): {inertia:.4f}")
    print("New supply locations:")
    for i, c in enumerate(new_c):
        if space == "euclidean":
            print(f"  new_{i}  (cluster {n_supply + i}):  x={c[0]:.4f}  y={c[1]:.4f}")
        else:
            print(f"  new_{i}  (cluster {n_supply + i}):  lon={c[0]:.6f}  lat={c[1]:.6f}")

    # ── save CSV ──────────────────────────────────────────────────────────────

    out_df                   = demand_df.copy()
    out_df["cluster_before"] = lbl_before
    out_df["cluster_after"]  = lbl_after
    out_df["is_new_cluster"] = lbl_after >= n_supply
    if space == "network" and snaps_demand is not None:
        out_df["snap_lon"] = [s.snap_lon for s in snaps_demand]
        out_df["snap_lat"] = [s.snap_lat for s in snaps_demand]
    out_df.to_csv(output, index=False)
    print(f"Saved assignments → {output}")

    # ── plot ──────────────────────────────────────────────────────────────────

    _plot(demand_xy, fixed_xy, new_xy, lbl_before, lbl_after,
          figure, space, G=G_plot)


if __name__ == "__main__":
    main()
