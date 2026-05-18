"""Generate synthetic demand.csv and supply.csv for testing kmeans_pp.py.

Creates 5 demand blobs in 2D. Existing supply covers 3 of them; the other 2
are intentionally uncovered, so running kmeans_pp.py with --n-new 2 should
place new centers near those uncovered blobs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def make_blobs(
    centers: np.ndarray, n_samples: int, std: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Tiny numpy-only replacement for sklearn.datasets.make_blobs."""
    k = len(centers)
    blob_id = rng.integers(0, k, size=n_samples)
    X = centers[blob_id] + rng.normal(0.0, std, size=(n_samples, centers.shape[1]))
    return X, blob_id


def main() -> None:
    rng = np.random.default_rng(42)

    blob_centers = np.array([
        [0.0, 0.0],
        [10.0, 0.0],
        [0.0, 10.0],
        [10.0, 10.0],
        [5.0, 5.0],
    ])

    X, blob_id = make_blobs(blob_centers, n_samples=500, std=0.8, rng=rng)
    weights = rng.integers(1, 11, size=len(X)).astype(float)

    demand = pd.DataFrame({
        "id": [f"D{i:04d}" for i in range(len(X))],
        "x": X[:, 0],
        "y": X[:, 1],
        "weight": weights,
        "source_blob": blob_id,
    })

    # Existing supply covers blobs 0, 1, 2 — blobs 3 and 4 are uncovered.
    supply = pd.DataFrame({
        "id": ["S0", "S1", "S2"],
        "name": ["depot_A", "depot_B", "depot_C"],
        "x": [0.2, 10.1, -0.3],
        "y": [-0.1, 0.4, 9.8],
    })

    out_dir = Path(__file__).parent
    demand.to_csv(out_dir / "demand.csv", index=False)
    supply.to_csv(out_dir / "supply.csv", index=False)
    print(f"Wrote {out_dir / 'demand.csv'} ({len(demand)} rows)")
    print(f"Wrote {out_dir / 'supply.csv'} ({len(supply)} rows)")
    print("Try: python kmeans_pp.py --demand demand.csv --supply supply.csv --n-new 2")


if __name__ == "__main__":
    main()
