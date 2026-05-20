# Location Allocation — Constrained K-Means++

Place **N new supply facilities** optimally among existing demand points, without moving existing supply. Supports three distance spaces: raw Euclidean, projected metric (UTM), and real-world street-network distance via OpenStreetMap.

---

## Features

- **K-Means++ seeding** — probabilistic initialisation for consistently good solutions
- **Constrained Lloyd iteration** — existing supply centres are frozen; only new centres move
- **Three distance spaces** — choose the one that matches your data
- **Edge snapping** *(network mode)* — demand and supply points are snapped to the nearest OSM street segment; new centres are placed on the network, not in free space
- **Before / after visualisation** — side-by-side map saved automatically
- **Multiple restarts** (`N_INIT`) — best solution across runs is kept
- **Config-driven** — all parameters are plain variables at the top of each script; no CLI flags needed

---

## Distance Spaces

| `SPACE` | How distance is measured | Coordinate input | Extra deps |
|---------|--------------------------|------------------|------------|
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
| `y` | yes | y-coordinate (or **latitude** for geometric/network) |
| `weight` | no | demand weight (defaults to 1 if omitted or `WEIGHT_COL = ""`) |
| any others | no | preserved as-is in the output CSV |

### `supply.csv`

| Column | Required | Description |
|--------|----------|-------------|
| `x` | yes | x-coordinate (or longitude) |
| `y` | yes | y-coordinate (or latitude) |
| any others | no | ignored |

---

## Usage

Edit the config block at the top of `kmeans_pp.py`, then run:

```bash
python kmeans_pp.py
```

### Configuration variables (`kmeans_pp.py`)

| Variable | Default | Description |
|----------|---------|-------------|
| `DEMAND` | `demand.csv` | Path to demand CSV |
| `SUPPLY` | `supply.csv` | Path to existing supply CSV |
| `N_NEW` | `2` | Number of new supply centres to add |
| `SPACE` | `euclidean` | Distance space: `euclidean` / `geometric` / `network` |
| `WEIGHT_COL` | `weight` | Column in demand CSV to use as weight; set to `""` to ignore |
| `NETWORK_TYPE` | `drive` | OSM network type: `drive` / `walk` / `bike` / `all` |
| `BUFFER` | `500` | Extra metres to pad the OSM bounding box |
| `N_INIT` | `10` | Number of random restarts (best result kept) |
| `MAX_ITER` | `300` | Max Lloyd iterations per restart |
| `SEED` | `0` | Random seed for reproducibility |
| `OUTPUT` | `demand_clustered.csv` | Path for demand assignments output CSV |
| `CENTRES_OUTPUT` | `new_centres.csv` | Path for new supply centres output CSV |
| `FIGURE` | `clusters_before_after.png` | Path for output figure |

---

## Output

### Demand assignments (`demand_clustered.csv`)

Original demand columns plus:

| Column | Description |
|--------|-------------|
| `cluster_before` | Cluster index assigned to each demand point using existing supply only |
| `cluster_after` | Cluster index after adding new supply (indices `>= n_existing` are new) |
| `is_new_cluster` | `True` if the demand point is now served by a new centre |
| `snap_lon` | *(network mode only)* Longitude of the snapped demand position on the network |
| `snap_lat` | *(network mode only)* Latitude of the snapped demand position on the network |

### New supply centres (`new_centres.csv`)

One row per new supply centre:

| Column | Description |
|--------|-------------|
| `cluster_id` | Cluster index this centre represents (always `>= n_existing`) |
| `x`, `y` | Coordinates *(euclidean mode only)* |
| `lon`, `lat` | Coordinates *(geometric / network mode only)* |
| `n_demand` | Number of demand points assigned to this centre after optimisation |
| `total_weight` | Sum of demand weights assigned to this centre |
| `mean_weight` | Average demand weight per assigned point |

### Figure (`clusters_before_after.png`)

Two side-by-side panels:
- **Before** — demand coloured by cluster, existing supply shown as black squares
- **After** — same, with new supply added as red stars; network edges drawn as grey background in network mode

---

## Quick-start with demo data

### 1. Configure and generate demo data

Edit the config block at the top of `make_demo_data.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `BLOB_CENTERS` | 5 blobs | List of `[x, y]` blob centre coordinates |
| `N_SAMPLES` | `500` | Total demand points spread across all blobs |
| `BLOB_STD` | `0.8` | Spread (standard deviation) of each blob |
| `WEIGHT_MIN` / `WEIGHT_MAX` | `1` / `10` | Range for random demand weights |
| `SEED` | `42` | Random seed |
| `SUPPLY_OFFSETS` | 3 offsets | One `[dx, dy]` per covered blob; shorten to leave more blobs uncovered |
| `SUPPLY_NAMES` | `depot_A/B/C` | Names for each supply depot |
| `DEMAND_OUTPUT` | `demand.csv` | Output demand CSV filename |
| `SUPPLY_OUTPUT` | `supply.csv` | Output supply CSV filename |

Then run:

```bash
python make_demo_data.py
```

The script prints the suggested `N_NEW` value to use in `kmeans_pp.py`.

### 2. Run the analysis

```bash
python kmeans_pp.py
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
3. Repeat until convergence (`MAX_ITER` cap) or centre positions stop changing.

The best result across `N_INIT` independent restarts is returned.

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
