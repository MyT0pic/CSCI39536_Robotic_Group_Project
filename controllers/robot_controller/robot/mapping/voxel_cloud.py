"""Point cloud with at most one point per cube."""
from typing import Optional

import numpy as np

from robot.utils.geometry import voxel_keys


class VoxelCloud:
    """Point cloud with at most one point per voxel_size cube; with ranges, a cube seen again from closer takes the closer point."""

    def __init__(self, voxel_size: float):
        self.voxel_size = voxel_size
        # Points live in buffers with room to grow (float32: plenty for mm-level map coordinates and half
        # the memory), so adding a keyframe doesn't copy the whole map every time
        self._points = np.empty((1024, 3), dtype=np.float32)
        self._ranges = np.empty(1024, dtype=np.float32)
        self._count = 0
        self._keys = np.empty(0, dtype=np.int64)  # sorted keys of the filled cubes ...
        self._rows = np.empty(0, dtype=np.int64)  # ... and the row of points holding each one
        self.replaced = np.empty(0, dtype=np.int64)  # rows whose point the last add() replaced with a closer one

    def add(self, points: np.ndarray, ranges: Optional[np.ndarray] = None) -> np.ndarray:
        """Add points (with their sensor ranges); returns the rows of points that filled new cubes."""
        self.replaced = np.empty(0, dtype=np.int64)
        if len(points) == 0:
            return np.empty(0, dtype=np.int64)
        # One point per cube (the average), and the closest range that saw it
        keys, inverse, counts = np.unique(voxel_keys(points, self.voxel_size), return_inverse=True, return_counts=True)
        points = np.column_stack([np.bincount(inverse, weights=points[:, k], minlength=len(keys)) for k in range(3)]) / counts[:, None]
        closest = np.full(len(keys), np.inf)
        if ranges is not None:
            np.minimum.at(closest, inverse, ranges)

        slot = np.searchsorted(self._keys, keys)
        known = np.zeros(len(keys), dtype=bool)
        inside = slot < len(self._keys)
        known[inside] = self._keys[slot[inside]] == keys[inside]

        # Cubes already in the cloud: take the new point if it was seen from closer
        rows = self._rows[slot[known]]
        closer = closest[known] < self._ranges[rows]
        self._points[rows[closer]] = points[known][closer]
        self.replaced = rows[closer]
        self._ranges[rows[closer]] = closest[known][closer]

        # New cubes
        start, added = self._count, int((~known).sum())
        self._reserve(start + added)
        self._points[start:start + added] = points[~known]
        self._ranges[start:start + added] = closest[~known]
        self._count = start + added
        new_rows = np.arange(start, self._count)
        where = np.searchsorted(self._keys, keys[~known])
        self._keys = np.insert(self._keys, where, keys[~known])
        self._rows = np.insert(self._rows, where, new_rows)
        return new_rows

    @property
    def points(self) -> np.ndarray:
        """(N, 3) float32 points (a view: copy it to keep it past the next add)."""
        return self._points[:self._count]

    @property
    def ranges(self) -> np.ndarray:
        """Closest sensor distance of each point (inf if not given)."""
        return self._ranges[:self._count]

    def _reserve(self, size: int) -> None:
        if size <= len(self._points):
            return
        capacity = max(size, 2 * len(self._points))
        points, ranges = np.empty((capacity, 3), dtype=np.float32), np.empty(capacity, dtype=np.float32)
        points[:self._count], ranges[:self._count] = self.points, self.ranges
        self._points, self._ranges = points, ranges
