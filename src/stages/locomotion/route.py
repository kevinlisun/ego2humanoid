"""L.13: grid A* over the scenario occupancy map -> (x, y, heading) waypoints for the goto executor."""

from __future__ import annotations

import heapq

import numpy as np

SQRT2 = 2**0.5
NEIGHBORS = [
    (1, 0, 1.0),
    (-1, 0, 1.0),
    (0, 1, 1.0),
    (0, -1, 1.0),
    (1, 1, SQRT2),
    (1, -1, SQRT2),
    (-1, 1, SQRT2),
    (-1, -1, SQRT2),
]


def _cell(p, extent, res):
    return int((p[1] - extent[1][0]) / res), int((p[0] - extent[0][0]) / res)


def plan_route(
    grid: np.ndarray,
    extent,
    res: float,
    start_xy,
    goal_xyyaw,
    spacing: float = 0.5,
) -> list[tuple[float, float, float]] | None:
    """A* on a blocked-cell grid (already inflated by the base radius); None if no path."""
    s, g = _cell(start_xy, extent, res), _cell(goal_xyyaw[:2], extent, res)
    h, w = grid.shape
    if not (0 <= s[0] < h and 0 <= s[1] < w and 0 <= g[0] < h and 0 <= g[1] < w) or grid[s] or grid[g]:
        return None
    dist, prev, heap = {s: 0.0}, {}, [(0.0, s)]
    while heap:
        _, c = heapq.heappop(heap)
        if c == g:
            break
        for dr, dc, cost in NEIGHBORS:
            n = (c[0] + dr, c[1] + dc)
            if not (0 <= n[0] < h and 0 <= n[1] < w) or grid[n]:
                continue
            if dr and dc and (grid[c[0] + dr, c[1]] or grid[c[0], c[1] + dc]):
                continue  # no corner cutting
            nd = dist[c] + cost
            if nd < dist.get(n, np.inf):
                dist[n], prev[n] = nd, c
                heapq.heappush(heap, (nd + float(np.hypot(g[0] - n[0], g[1] - n[1])), n))
    if g not in dist:
        return None
    cells = [g]
    while cells[-1] != s:
        cells.append(prev[cells[-1]])
    pts = [(extent[0][0] + (c[1] + 0.5) * res, extent[1][0] + (c[0] + 0.5) * res) for c in reversed(cells)]
    keep, last = [], np.array(pts[0])
    for p in pts[1:-1]:
        if np.linalg.norm(np.array(p) - last) >= spacing:
            keep.append(p)
            last = np.array(p)
    wps = [(x, y, 0.0) for x, y in keep] + [tuple(goal_xyyaw)]
    return [
        (x, y, float(np.arctan2(wps[i + 1][1] - y, wps[i + 1][0] - x)) if i < len(wps) - 1 else yaw)
        for i, (x, y, yaw) in enumerate(wps)
    ]
