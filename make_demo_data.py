"""Generate synthetic demand.csv and supply.csv for testing kmeans_pp.py.

Creates demand blobs in 2D. Existing supply covers some blobs; uncovered blobs
are left intentionally so kmeans_pp.py can place new centres there.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ── CONFIGURATION ─────────────────────────────────────────────────────────────

# Demand blob centres — each row is [x, y].
# The first N_SUPPLY_BLOBS blobs will have a matching supply depot placed near
# them; the remaining blobs are left uncovered (good targets for new supply).
BLOB_CENTERS = [
    [0.0,  0.0],
    [10.0, 0.0],
    [0.0,  10.0],
    [10.0, 10.0],
    [5.0,  5.0],
]

N_SAMPLES    = 500    # total demand points spread across all blobs
BLOB_STD     = 0.8    # standard deviation (spread) of each blob
WEIGHT_MIN   = 1      # minimum demand weight per point
WEIGHT_MAX   = 10     # maximum demand weight per point (inclusive)
SEED         = 42     # random seed for reproducibility

# Supply depots — one per covered blob; leave the rest uncovered.
# Each entry is [x_offset, y_offset] relative to the matching blob centre.
SUPPLY_OFFSETS = [
    [ 0.2, -0.1],   # near blob 0
    [ 0.1,  0.4],   # near blob 1
    [-0.3, -0.2],   # near blob 2
]
SUPPLY_ID_PREFIX   = "S"
SUPPLY_NAME_PREFIX = "depot_"
SUPPLY_NAMES       = ["depot_A", "depot_B", "depot_C"]  # one per supply offset

DEMAND_OUTPUT = "demand.csv"
SUPPLY_OUTPUT = "supply.csv"

# ─────────────────────────────────────────────────────────────────────────────


def make_blobs(
    centers: np.ndarray, n_samples: int, std: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    k = len(centers)
    blob_id = rng.integers(0, k, size=n_samples)
    X = centers[blob_id] + rng.normal(0.0, std, size=(n_samples, centers.shape[1]))
    return X, blob_id


def main() -> None:
    rng          = np.random.default_rng(SEED)
    blob_centers = np.array(BLOB_CENTERS, dtype=float)

    X, blob_id = make_blobs(blob_centers, n_samples=N_SAMPLES, std=BLOB_STD, rng=rng)
    weights    = rng.integers(WEIGHT_MIN, WEIGHT_MAX + 1, size=len(X)).astype(float)

    demand = pd.DataFrame({
        "id":          [f"D{i:04d}" for i in range(len(X))],
        "x":           X[:, 0],
        "y":           X[:, 1],
        "weight":      weights,
        "source_blob": blob_id,
    })

    offsets      = np.array(SUPPLY_OFFSETS, dtype=float)
    supply_xy    = blob_centers[:len(offsets)] + offsets
    n_supply     = len(supply_xy)
    supply_names = list(SUPPLY_NAMES)[:n_supply]
    while len(supply_names) < n_supply:
        supply_names.append(f"{SUPPLY_NAME_PREFIX}{len(supply_names)}")

    supply = pd.DataFrame({
        "id":   [f"{SUPPLY_ID_PREFIX}{i}" for i in range(n_supply)],
        "name": supply_names,
        "x":    supply_xy[:, 0],
        "y":    supply_xy[:, 1],
    })

    out_dir = Path(__file__).parent
    demand.to_csv(out_dir / DEMAND_OUTPUT, index=False)
    supply.to_csv(out_dir / SUPPLY_OUTPUT, index=False)
    n_uncovered = len(blob_centers) - n_supply
    print(f"Wrote {out_dir / DEMAND_OUTPUT} ({len(demand)} rows)")
    print(f"Wrote {out_dir / SUPPLY_OUTPUT} ({n_supply} depots, {n_uncovered} blobs uncovered)")
    print(f"Suggested N_NEW in kmeans_pp.py: {n_uncovered}")


if __name__ == "__main__":
    main()
