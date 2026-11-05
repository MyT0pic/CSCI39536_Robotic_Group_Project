"""Path planning on a cost grid: Dijkstra search, then straight lines to drive."""
import math
from typing import List, Tuple
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from robot.utils.types import Cell


_MOVES = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]


def grid_graph(cost: np.ndarray):
    """Sparse 8-connected graph of a cost grid: a move costs its length (cells) times the mean cost of its two cells; infinite-cost cells are left out."""
    flat = np.isfinite(cost).reshape(-1)
    weights = np.where(np.isfinite(cost), cost, 0.0)
    rows, cols, values = [], [], []
    index = np.arange(cost.size).reshape(cost.shape)
    nx, ny = cost.shape
    for di, dj in _MOVES:
        # cells (i, j) whose neighbor (i + di, j + dj) is inside the grid
        a = index[max(0, -di):nx - max(0, di), max(0, -dj):ny - max(0, dj)].reshape(-1)
        b = index[max(0, di):nx - max(0, -di), max(0, dj):ny - max(0, -dj)].reshape(-1)
        ok = flat[a] & flat[b]
        a, b = a[ok], b[ok]
        rows.append(a)
        cols.append(b)
        values.append(math.hypot(di, dj) * (weights.flat[a] + weights.flat[b]) / 2)
    return coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(cost.size, cost.size)).tocsr()


class PathSearch:
    """Cheapest paths from one start cell to every cell (one Dijkstra search)."""

    def __init__(self, cost: np.ndarray, start: Cell):
        self.shape = cost.shape
        self.start = start
        distance, predecessors = dijkstra(grid_graph(cost), indices=np.ravel_multi_index(start, cost.shape),
                                          return_predecessors=True)
        self.distance = distance.reshape(cost.shape)  # cost of the cheapest way to each cell, inf if none
        self._predecessors = predecessors

    def reachable(self, cell: Cell) -> bool:
        """Whether cell can be reached."""
        return bool(np.isfinite(self.distance[cell]))

    def path_to(self, goal: Cell) -> List[Cell]:
        """Cells from the start to goal, both included; empty if unreachable."""
        if not self.reachable(goal):
            return []
        node = int(np.ravel_multi_index(goal, self.shape))
        start = int(np.ravel_multi_index(self.start, self.shape))
        nodes = [node]
        while node != start:
            node = int(self._predecessors[node])
            nodes.append(node)
        i, j = np.unravel_index(np.array(nodes[::-1]), self.shape)
        return list(zip(i.tolist(), j.tolist()))


def line_cells(ends: List[Cell]) -> List[Cell]:
    """Cells along the straight lines between ends, one step apart."""
    out = [tuple(ends[0])] if ends else []
    for a, b in zip(ends[:-1], ends[1:]):
        i, j = _line(np.array(a), np.array(b))
        out.extend(list(zip(i.tolist(), j.tolist()))[1:])
    return out


def corners(cells: List[Cell], cost: np.ndarray) -> List[Cell]:
    """Straighten a grid path: the ends of straight lines that cross no costlier cells than the path does there."""
    if len(cells) < 3:
        return list(cells)
    path = np.array(cells)
    path_cost = cost[path[:, 0], path[:, 1]]
    kept = [0]
    i = 0
    while i < len(path) - 1:
        j = i + 1
        # (the cell the robot stands on is left out of the limit: it may carry a high escape cost)
        while j + 1 < len(path) and _clear_line(path[i], path[j + 1], cost, path_cost[max(i, 1):j + 2].max()):
            j += 1
        kept.append(j)
        i = j
    return [tuple(path[k]) for k in kept]


def along(points: np.ndarray, step: float = 0.1) -> np.ndarray:
    """Points every step (m) along the straight lines between points (N, 2), ends included."""
    if len(points) < 2:
        return np.asarray(points, dtype=float).reshape(-1, 2)
    out = [points[:1]]
    for a, b in zip(points[:-1], points[1:]):
        n = max(1, int(np.ceil(np.hypot(*(b - a)) / step)))
        out.append(a + (b - a) * (np.arange(1, n + 1) / n)[:, None])
    return np.vstack(out)


def _line(a: np.ndarray, b: np.ndarray, per_cell: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Cells along the line from a to b, per_cell samples per cell of length."""
    steps = int(per_cell * max(abs(b[0] - a[0]), abs(b[1] - a[1]))) + per_cell - 1
    t = np.linspace(0.0, 1.0, steps + 1)
    return np.rint(a[0] + (b[0] - a[0]) * t).astype(int), np.rint(a[1] + (b[1] - a[1]) * t).astype(int)


def _clear_line(a: np.ndarray, b: np.ndarray, cost: np.ndarray, limit: float) -> bool:
    """Whether every cell on the line from a to b is passable and costs at most limit."""
    c = cost[_line(a, b, per_cell=2)]
    return bool(np.all(np.isfinite(c)) and np.all(c <= limit + 1e-9))
