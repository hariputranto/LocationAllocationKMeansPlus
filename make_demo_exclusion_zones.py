"""Generate exclusion_zones.geojson for testing kmeans_pp.py.

Defines polygon areas where new supply centres are NOT allowed to be placed.
Coordinates must match the input space used in kmeans_pp.py:
  - x/y values for SPACE = "euclidean"
  - lon/lat values for SPACE = "geometric" or "network"

Each zone is defined by its vertices as [x, y] pairs (polygon auto-closed).
Use the rect() helper below to build rectangular zones quickly.
"""

from __future__ import annotations

import json
from pathlib import Path


# ── CONFIGURATION ─────────────────────────────────────────────────────────────

# Each zone is a dict with:
#   "name"     — label shown in the GeoJSON properties (informational)
#   "vertices" — list of [x, y] corner points; polygon is auto-closed
#
# Tip: use rect(x_min, y_min, x_max, y_max) to define a rectangle easily.
# The demo zones below cover blobs 3 ([10, 10]) and 4 ([5, 5]) of the default
# make_demo_data.py output, forcing new centres away from those areas.

ZONES = [
    {
        "name": "restricted_area_1",
        "vertices": [
            [9.0,  9.0],
            [11.5, 9.0],
            [11.5, 11.5],
            [9.0,  11.5],
        ],
    },
    {
        "name": "restricted_area_2",
        "vertices": [
            [4.0, 4.0],
            [6.0, 4.0],
            [6.0, 6.0],
            [4.0, 6.0],
        ],
    },
]

OUTPUT = "exclusion_zones.geojson"

# ─────────────────────────────────────────────────────────────────────────────


def rect(x_min: float, y_min: float, x_max: float, y_max: float) -> list[list[float]]:
    """Return four corner vertices for a rectangular zone."""
    return [
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max],
    ]


def _build_feature(name: str, vertices: list[list[float]]) -> dict:
    ring = [list(v) for v in vertices]
    if ring[0] != ring[-1]:
        ring.append(ring[0])   # auto-close
    return {
        "type": "Feature",
        "properties": {"name": name},
        "geometry": {
            "type": "Polygon",
            "coordinates": [ring],
        },
    }


def main() -> None:
    if not ZONES:
        raise SystemExit("ZONES list is empty — add at least one zone.")

    features = [_build_feature(z["name"], z["vertices"]) for z in ZONES]
    geojson  = {"type": "FeatureCollection", "features": features}

    out_path = Path(__file__).parent / OUTPUT
    with open(out_path, "w") as f:
        json.dump(geojson, f, indent=2)

    print(f"Wrote {out_path} ({len(features)} zone(s))")
    for z in ZONES:
        verts = z["vertices"]
        xs    = [v[0] for v in verts]
        ys    = [v[1] for v in verts]
        print(f"  {z['name']}: x=[{min(xs)}, {max(xs)}]  y=[{min(ys)}, {max(ys)}]")
    print(f"\nTo enable in kmeans_pp.py, set:  EXCLUSION_ZONES = \"{OUTPUT}\"")


if __name__ == "__main__":
    main()
