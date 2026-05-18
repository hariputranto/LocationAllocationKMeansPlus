# Location Allocation — Constrained K-Means++

Place **N new supply facilities** optimally among existing demand points, without moving existing supply. Supports three distance spaces: raw Euclidean, projected metric (UTM), and real-world street-network distance via OpenStreetMap.

---

## Features

- **K-Means++ seeding** — probabilistic initialisation for consistently good solutions
- **Constrained Lloyd iteration** — existing supply centres are frozen; only new centres move
- **Three distance spaces** — choose the one that matches your data
- **Edge snapping** *(network mode)* — demand and supply points are snapped to the nearest OSM street segment; new centres are placed on the network, not in free space
- **Before / after visualisation** — side-by-side map saved automatically
- **Multiple restarts** (`--n-init`) — best solution across runs is kept

---

## Distance Spaces

| `--space` | How distance is measured | Coordinate input | Extra deps |
|-----------|--------------------------|------------------|------------|
| `euclidean` | Straight-line in x/y | any units | — |
| `geometric` | Straight-line in **metres** (UTM projected) | longitude / latitude | `pyproj` |
| `network` | **Shortest-path** along OSM streets in metres | longitude / latitude | `osmnx networkx shapely pyproj` |

### Edge snapping (network mode)

Each point is orthogonally projected onto its nearest OSM edge (the perpendicular foot-point). The snapped position retains full network connectivity: distances to both edge endpoints are recorded and used when computing shortest-path distances through the graph. New supply locations are returned at the snapped edge position of the optimal demand medoid in each cluster.

---

## Installation

```bash
# clone / download the repo, then:
pip install -r requirements.txt
```

Install only what you need:

```bash
# euclidean mode only
pip install numpy pandas matplotlib

# add geometric mode
pip install pyproj

# add network mode
pip install osmnx networkx shapely pyproj
```

---

## Input CSV Format

### `demand.csv`

| Column | Required | Description |
|--------|----------|-------------|
| `x` | yes | x-coordinate (or **longitude** for geometric/network) |
| `y` | yes | y-coordinate (or **latitude**  for geometric/network) |
| `weight` | no | demand weight (defaults to 1 if omitted) |
| any others | no | preserved as-is in the output CSV |

### `supply.csv`

| Column | Required | Description |
|--------|----------|-------------|
| `x` | yes | x-coordinate (or longitude) |
| `y` | yes | y-coordinate (or latitude) |
| any others | no | ignored |

---

## Usage

```bash
# Euclidean space (plain x/y coordinates)
python kmeans_pp.py --demand demand.csv --supply supply.csv --n-new 2

# Geometric space (lon/lat → UTM, distances in metres)
python kmeans_pp.py --demand demand.csv --supply supply.csv --n-new 2 \
    --space geometric

# Network space (OSM street network, distances in metres)
python kmeans_pp.py --demand demand.csv --supply supply.csv --n-new 2 \
    --space network

# Network space — walking network, 1 km OSM buffer
python kmeans_pp.py --demand demand.csv --supply supply.csv --n-new 3 \
    --space network --network-type walk --buffer 1000
```

### All arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--demand` | *(required)* | Path to demand CSV |
| `--supply` | *(required)* | Path to existing supply CSV |
| `--n-new` | *(required)* | Number of new supply centres to add |
| `--space` | `euclidean` | Distance space: `euclidean` / `geometric` / `network` |
| `--network-type` | `drive` | OSMnx network type: `drive` / `walk` / `bike` / `all` |
| `--buffer` | `500` | Extra metres to pad the OSM bounding box |
| `--n-init` | `10` | Number of random restarts (best result kept) |
| `--max-iter` | `300` | Max Lloyd iterations per restart |
| `--seed` | `0` | Random seed for reproducibility |
| `--output` | `demand_clustered.csv` | Path for output CSV |
| `--figure` | `clusters_before_after.png` | Path for output figure |

---

## Output

### CSV (`demand_clustered.csv`)

Original demand columns plus:

| Column | Description |
|--------|-------------|
| `cluster_before` | Cluster index assigned to each demand point using existing supply only |
| `cluster_after` | Cluster index after adding new supply (indices `>= n_existing` are new) |
| `is_new_cluster` | `True` if the demand point is now served by a new centre |
| `snap_lon` | *(network mode only)* Longitude of the snapped demand position on the network |
| `snap_lat` | *(network mode only)* Latitude  of the snapped demand position on the network |

### Figure (`clusters_before_after.png`)

Two side-by-side panels:
- **Before** — demand coloured by cluster, existing supply shown as black squares
- **After** — same, with new supply added as red stars; network edges drawn as grey background in network mode

---

## Quick-start with demo data

```bash
# Generate synthetic demand (500 points, 5 blobs) and supply (3 existing depots)
python make_demo_data.py

# Run — existing supply covers 3 blobs; add 2 new centres for the uncovered blobs
python kmeans_pp.py --demand demand.csv --supply supply.csv --n-new 2
```

---

## Algorithm

### K-Means++ seeding

New centres are chosen sequentially. At each step the probability of selecting demand point **x** is proportional to:

```
P(x) ∝ weight(x) · D(x)²
```

where `D(x)` is the distance from **x** to the nearest already-placed centre (existing supply or previously seeded new centre).

### Constrained Lloyd iteration

1. **Assign** each demand point to its nearest centre (existing or new).
2. **Update** only the new centres:
   - *Euclidean / geometric*: weighted mean of assigned demand coordinates.
   - *Network*: weighted **medoid** — the assigned demand point that minimises `Σ w_j · d(i, j)` over all cluster members.
3. Repeat until convergence (`--max-iter` cap) or centre positions stop changing.

The best result across `--n-init` independent restarts is returned.

### Network distance computation

For two points snapped to edges (u₁, v₁) and (u₂, v₂) at foot-distances `d_u1`, `d_v1`, `d_u2`, `d_v2`:

```
dist(P₁, P₂) = min(
    d_u1 + sp(u₁, u₂) + d_u2,
    d_u1 + sp(u₁, v₂) + d_v2,
    d_v1 + sp(v₁, u₂) + d_u2,
    d_v1 + sp(v₁, v₂) + d_v2,
    |d_u1 − d_u2|  ← same-edge shortcut
)
```

where `sp(·, ·)` is the pre-computed Dijkstra shortest-path length between graph nodes.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥ 1.24 | Array maths |
| pandas | ≥ 1.5 | CSV I/O |
| matplotlib | ≥ 3.6 | Visualisation |
| pyproj | ≥ 3.4 | UTM projection (geometric + network) |
| osmnx | ≥ 1.6 | OSM network download + edge snapping |
| networkx | ≥ 3.0 | Shortest-path computation |
| shapely | ≥ 2.0 | Edge geometry / foot-point projection |
