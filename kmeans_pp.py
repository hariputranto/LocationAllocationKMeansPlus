"""Constrained K-Means with K-Means++ seeding for new supply locations.

Problem
-------
You have demand points and existing supply points (frozen cluster centers).
You want to add N new supply points so that demand is served as efficiently
as possible, without moving the existing centers.

Algorithm
---------
1. K-Means++ seeding for the N new centers: D(x) is the distance from a demand
   point to the nearest already-placed center (existing supply OR a new center
   chosen earlier in seeding). Sampling probability is proportional to
   weight(x) * D(x)^2.
2. Constrained Lloyd's iteration: assign each demand to its nearest center
   (existing or new), then update ONLY the new centers as the weighted mean
   of demands assigned to them. Existing centers never move.

Inputs (CSV)
------------
--demand demand.csv  : must contain columns 'x', 'y'. Optional 'weight' column
                       (defaults to 1). All other columns are preserved in the
                       output CSV.
--supply supply.csv  : must contain columns 'x', 'y'.

Outputs
-------
- A figure with two panels: cluster assignment before vs. after adding new
  supply, with existing and new centers marked.
- A CSV: the original demand rows plus 'cluster_before', 'cluster_after',
  and 'is_new_cluster' columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------- Core algorithm -------------------------------------------------


def _nearest_sq_dist(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """For each row of X, squared distance to the nearest row of centers."""
    # (n, k) matrix of squared distances, then min over k.
    diff = X[:, None, :] - centers[None, :, :]
    return np.min(np.sum(diff ** 2, axis=2), axis=1)


def kmeans_pp_init_with_fixed(
    X: np.ndarray,
    weights: np.ndarray,
    fixed_centers: np.ndarray,
    n_new: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pick n_new centers from X via K-Means++, treating fixed_centers as already-chosen."""
    n_samples, n_features = X.shape
    new_centers = np.empty((n_new, n_features), dtype=X.dtype)

    sq_dist = _nearest_sq_dist(X, fixed_centers)

    for i in range(n_new):
        scores = weights * sq_dist
        total = scores.sum()
        if total <= 0:
            # Every demand coincides with a center — pick uniformly at random.
            idx = rng.integers(n_samples)
        else:
            idx = rng.choice(n_samples, p=scores / total)
        new_centers[i] = X[idx]

        # Update D(x)^2 to include the newly chosen center.
        new_sq = np.sum((X - new_centers[i]) ** 2, axis=1)
        sq_dist = np.minimum(sq_dist, new_sq)

    return new_centers


def assign(X: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (labels, squared-distance-to-assigned-center) for each row of X."""
    diff = X[:, None, :] - centers[None, :, :]
    sq = np.sum(diff ** 2, axis=2)
    labels = np.argmin(sq, axis=1)
    return labels, sq[np.arange(len(X)), labels]


def constrained_kmeans(
    X: np.ndarray,
    weights: np.ndarray,
    fixed_centers: np.ndarray,
    n_new: int,
    max_iter: int = 300,
    tol: float = 1e-6,
    n_init: int = 10,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run constrained K-Means; only the n_new new centers move.

    Returns (new_centers, labels, inertia), where labels index into the
    concatenated (fixed_centers, new_centers) array.
    """
    rng = np.random.default_rng(random_state)
    n_fixed = len(fixed_centers)
    best: tuple[np.ndarray, np.ndarray, float] | None = None

    for _ in range(n_init):
        new_centers = kmeans_pp_init_with_fixed(
            X, weights, fixed_centers, n_new, rng
        )
        prev_inertia = np.inf

        for _ in range(max_iter):
            all_centers = np.vstack([fixed_centers, new_centers])
            labels, sq_to_assigned = assign(X, all_centers)
            inertia = float((weights * sq_to_assigned).sum())

            updated = new_centers.copy()
            for j in range(n_new):
                mask = labels == (n_fixed + j)
                if mask.any():
                    w = weights[mask]
                    updated[j] = (X[mask] * w[:, None]).sum(axis=0) / w.sum()
                else:
                    # Empty cluster: re-seed at the demand farthest from any center.
                    sq = _nearest_sq_dist(X, all_centers)
                    updated[j] = X[np.argmax(weights * sq)]

            shift = np.sum((updated - new_centers) ** 2)
            new_centers = updated
            if abs(prev_inertia - inertia) < tol or shift < tol:
                break
            prev_inertia = inertia

        all_centers = np.vstack([fixed_centers, new_centers])
        labels, sq_to_assigned = assign(X, all_centers)
        inertia = float((weights * sq_to_assigned).sum())
        if best is None or inertia < best[2]:
            best = (new_centers, labels, inertia)

    assert best is not None
    return best


# ---------- I/O & plotting -------------------------------------------------


def load_points(
    path: Path, x_col: str = "x", y_col: str = "y", weight_col: str = "weight"
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None]:
    df = pd.read_csv(path)
    for col in (x_col, y_col):
        if col not in df.columns:
            raise ValueError(f"{path}: missing required column '{col}'")
    coords = df[[x_col, y_col]].to_numpy(dtype=float)
    weights = (
        df[weight_col].to_numpy(dtype=float) if weight_col in df.columns else None
    )
    return df, coords, weights


def plot_before_after(
    X: np.ndarray,
    fixed_centers: np.ndarray,
    new_centers: np.ndarray,
    labels_before: np.ndarray,
    labels_after: np.ndarray,
    save_path: Path | None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    cmap = "tab20"

    axes[0].scatter(X[:, 0], X[:, 1], c=labels_before, s=18, cmap=cmap, alpha=0.7)
    axes[0].scatter(
        fixed_centers[:, 0], fixed_centers[:, 1],
        marker="s", s=200, c="black", edgecolors="white", linewidths=1.5,
        label="existing supply",
    )
    axes[0].set_title("Before — only existing supply")
    axes[0].legend(loc="best")

    axes[1].scatter(X[:, 0], X[:, 1], c=labels_after, s=18, cmap=cmap, alpha=0.7)
    axes[1].scatter(
        fixed_centers[:, 0], fixed_centers[:, 1],
        marker="s", s=200, c="black", edgecolors="white", linewidths=1.5,
        label="existing supply",
    )
    axes[1].scatter(
        new_centers[:, 0], new_centers[:, 1],
        marker="*", s=320, c="red", edgecolors="white", linewidths=1.5,
        label="new supply",
    )
    axes[1].set_title("After — existing + new supply")
    axes[1].legend(loc="best")

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        print(f"Saved figure to {save_path}")
    plt.show()


# ---------- CLI ------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--demand", type=Path, required=True, help="demand CSV (x,y[,weight,...])")
    parser.add_argument("--supply", type=Path, required=True, help="existing supply CSV (x,y,...)")
    parser.add_argument("--n-new", type=int, required=True, help="number of new centers to add")
    parser.add_argument("--output", type=Path, default=Path("demand_clustered.csv"))
    parser.add_argument("--figure", type=Path, default=Path("clusters_before_after.png"))
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    demand_df, X, weights = load_points(args.demand)
    if weights is None:
        weights = np.ones(len(X))
    _, fixed_centers, _ = load_points(args.supply)

    if args.n_new < 1:
        raise SystemExit("--n-new must be >= 1")

    # Before: assign each demand to its nearest *existing* supply.
    labels_before, _ = assign(X, fixed_centers)

    # Solve for new centers.
    new_centers, labels_after, inertia = constrained_kmeans(
        X, weights, fixed_centers, args.n_new,
        max_iter=args.max_iter, n_init=args.n_init, random_state=args.seed,
    )
    n_fixed = len(fixed_centers)

    print(f"Final inertia (weighted SSE): {inertia:.4f}")
    print("New supply locations:")
    for i, c in enumerate(new_centers):
        print(f"  new_{i} (cluster {n_fixed + i}): x={c[0]:.4f}, y={c[1]:.4f}")

    # Write CSV: preserve original demand columns, append cluster info.
    out_df = demand_df.copy()
    out_df["cluster_before"] = labels_before
    out_df["cluster_after"] = labels_after
    out_df["is_new_cluster"] = labels_after >= n_fixed
    out_df.to_csv(args.output, index=False)
    print(f"Saved cluster assignments to {args.output}")

    plot_before_after(
        X, fixed_centers, new_centers, labels_before, labels_after, args.figure
    )


if __name__ == "__main__":
    main()
