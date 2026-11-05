"""2.5D grid map from the SLAM point cloud: ground height and UNKNOWN / FREE / BLOCKED per cell.

A cell is BLOCKED when something stands between step_height and clearance above its ground, or the ground is
too steep. The cloud only grows, so each build places only the new points.
"""
import math
from typing import Iterable, Optional

import numpy as np
from scipy import ndimage

from robot.utils.numeric import nan_median, neighborhood
from robot.utils.types import Cell, Point


class GridMap:
    UNKNOWN, FREE, BLOCKED = 0, 1, 2

    cell_size = 0.1            # cell width (the local height map uses it too)
    step_height = 0.2          # bumps lower than this can be driven over
    clearance = 0.85           # things higher than this above the ground pass over the robot
    max_slope = math.radians(15)  # steeper ground is blocked ...
    lip = 0.04                 # ... allowing this much extra rise over slope_span cells (ledges, noise)
    slope_span = 2             # cells apart that ground heights are compared for steepness
    min_patch = 4              # a blocked patch smaller than this (cells) ...
    min_lone_hits = 3          # ... needs a cell with this many points in the way, or it's speckle
    sink_tolerance = 0.08      # points this far below the ground around them are ignored
    close_range = 3.5          # low obstacles and steep ground count only where seen from this close
    tall_height = 0.4          # things at least this high count at any range
    max_range = 12.0           # points seen from farther than this are left out
    margin = 2.0               # room left around the points when laying out the grid

    def __init__(self):
        self._first = np.zeros(2, dtype=np.int64)  # cell (0, 0) is world cell _first: cell edges are always at
                                                   # multiples of cell_size, wherever the grid starts
        self.ground = np.full((0, 0), np.nan)      # ground height per cell, nan if unknown
        self.top = np.full((0, 0), np.nan)         # height of the highest point above the ground per cell
        self.closest = np.full((0, 0), np.inf)     # closest distance (m) each cell was seen from, inf if unseen
        self.state = np.zeros((0, 0), dtype=np.uint8)
        # kept between builds (see the module notes)
        self._shape = (0, 0)
        self._count = 0                             # points placed so far ...
        self._flat = np.empty(0, dtype=np.int64)    # ... the flat index of each one's cell ...
        self._order = np.empty(0, dtype=np.int64)   # ... and their indices sorted by cell

    def build(self, points: np.ndarray) -> None:
        """Build the grid from a world point cloud: (N, 3), or (N, 4) with the distance each point was seen from.

        Pass the same growing cloud each time: only the new points are placed.
        """
        if len(points) == 0:
            return
        if points.shape[1] > 3:
            ranges = points[:, 3]
            points = points[:, :3]
        else:
            ranges = np.zeros(len(points))
        self._place(points)
        shape = self._shape
        cells = shape[0] * shape[1]
        flat, order = self._flat, self._order
        z = points[:, 2].astype(float)
        used = ranges <= self.max_range
        close = used & (ranges <= self.close_range)

        # Cells in sorted order: where each cell's run of points starts, and which cell it is
        sorted_flat = flat[order]
        starts = np.concatenate([[0], np.flatnonzero(np.diff(sorted_flat)) + 1])
        cell_ids = sorted_flat[starts]

        def per_cell(values, reduce, empty):
            out = np.full(cells, empty)
            out[cell_ids] = reduce.reduceat(values[order], starts)
            return out

        # Ground height: the median of the lowest points around each cell, so one noisy point
        # (common far from the robot, where the ground is seen at a shallow angle) can't fake
        # a step or an obstacle. Done twice: points well below the first estimate are stray
        # (seen from far away at a grazing angle) and would drag the ground down, making the
        # real ground next to them look like something standing on it.
        # Where a cell was seen up close, its ground comes from the close points only: far points hit the ground
        # at a grazing angle, where a degree of tilt puts them many cm low, and one of them as the "lowest point"
        # sinks the ground under the close ones, so plain bumps beside it look a step high (half the brown on the
        # open lanes was that). Far points still give the ground where nothing was seen close
        keep = used
        for _ in range(2):
            lowest = per_cell(np.where(keep, z, np.inf), np.minimum, np.inf)
            seen = np.isfinite(lowest)
            lowest_close = per_cell(np.where(keep & close, z, np.inf), np.minimum, np.inf)
            lowest = np.where(np.isfinite(lowest_close), lowest_close, lowest)
            ground = self._median_around(np.where(seen, lowest, np.nan).reshape(shape))
            with np.errstate(invalid="ignore"):
                keep = used & ~(z < ground.reshape(-1)[flat] - self.sink_tolerance)

        # Something in the robot's way: above a drivable bump, below the robot's clearance
        height = np.where(keep, z - ground.reshape(-1)[flat], np.nan)
        with np.errstate(invalid="ignore"):
            in_the_way = (height > self.step_height) & (height < self.clearance) & (close | (height >= self.tall_height))
        hits = np.bincount(flat[in_the_way], minlength=cells).reshape(shape)  # points in the way, per cell
        blocked = hits > 0

        # Too steep: ground rises or drops too much compared with the cells slope_span cells away.
        # Measuring over a few cells rather than next door keeps a few cm of map noise from
        # looking like a slope, while steps and the sides of rocks still stand out
        n = self.slope_span
        rise_limit = n * self.cell_size * math.tan(self.max_slope) + self.lip
        padded = np.pad(ground, n, constant_values=np.nan)
        seen_close = (np.bincount(flat[close], minlength=cells) > 0).reshape(shape)
        padded_close = np.pad(seen_close, n, constant_values=False)
        for dx, dy in ((n, 0), (-n, 0), (0, n), (0, -n)):
            neighbor = padded[n + dx:n + dx + shape[0], n + dy:n + dy + shape[1]]
            neighbor_close = padded_close[n + dx:n + dx + shape[0], n + dy:n + dy + shape[1]]
            with np.errstate(invalid="ignore"):
                blocked |= (np.abs(ground - neighbor) > rise_limit) & seen_close & neighbor_close

        # A small blocked patch backed by only a stray point or two is speckle, not an obstacle: on bumpy
        # ground a cell or two now and then reads a little too steep. Left in, every speck bends the paths
        # around it and the robot weaves. Real obstacles span more cells, a thin pole is hit by many beams up
        # its height, and small rocks are caught separately by the local height map
        patches, count = ndimage.label(blocked, structure=np.ones((3, 3)))
        if count:
            size = np.bincount(patches.reshape(-1), minlength=count + 1)
            most_hits = np.zeros(count + 1)
            np.maximum.at(most_hits, patches.reshape(-1), hits.reshape(-1))
            keep_patch = (size >= self.min_patch) | (most_hits >= self.min_lone_hits)
            keep_patch[0] = False
            blocked = keep_patch[patches]

        self.ground = ground
        top = per_cell(np.where(np.isfinite(height), height, -np.inf), np.maximum, -np.inf)
        self.top = np.where(np.isfinite(top), top, np.nan).reshape(shape)
        self.closest = per_cell(np.where(used, ranges, np.inf), np.minimum, np.inf).reshape(shape)
        self.state = np.where(seen.reshape(shape), np.where(blocked, self.BLOCKED, self.FREE), self.UNKNOWN).astype(np.uint8)

    def _place(self, points: np.ndarray) -> None:
        """Place the new points in their cells, keeping all points sorted by cell; re-lay the grid out when needed."""
        same_cloud = 0 < self._count <= len(points)
        if same_cloud:  # spot-check: earlier points still in the cells they were placed in
            check = np.linspace(0, self._count - 1, min(self._count, 64)).astype(int)
            checked = self.cells_of(points[check])
            same_cloud = bool(self.inside(checked).all() and
                              np.all(np.ravel_multi_index((checked[:, 0], checked[:, 1]), self._shape) == self._flat[check]))
        new = points[self._count:] if same_cloud else points
        if same_cloud and len(new):
            new_cells = self.cells_of(new)
            same_cloud = bool(self.inside(new_cells).all())
        if not same_cloud:
            self._first = np.floor((points[:, :2].min(axis=0).astype(float) - self.margin) / self.cell_size).astype(np.int64)
            last = np.floor((points[:, :2].max(axis=0).astype(float) + self.margin) / self.cell_size).astype(np.int64)
            self._shape = tuple(int(v) for v in last - self._first + 1)
            placed = self.cells_of(points)
            self._flat = np.ravel_multi_index((placed[:, 0], placed[:, 1]), self._shape)
            self._order = np.argsort(self._flat, kind="stable")
        elif len(new):
            new_flat = np.ravel_multi_index((new_cells[:, 0], new_cells[:, 1]), self._shape)
            new_order = np.argsort(new_flat, kind="stable")
            where = np.searchsorted(self._flat[self._order], new_flat[new_order], side="right")
            self._order = np.insert(self._order, where, new_order + self._count)
            self._flat = np.concatenate([self._flat, new_flat])
        self._count = len(points)

    @staticmethod
    def _median_around(heights: np.ndarray) -> np.ndarray:
        """Median of each known cell and its 8 neighbors (nan stays nan)."""
        median = nan_median(neighborhood(heights, 1))
        return np.where(np.isnan(heights), np.nan, median)

    def set_layers(self, first: np.ndarray, ground: np.ndarray, top: np.ndarray, blocked: np.ndarray) -> None:
        """Set the layers directly instead of building them from points (used by the local height map).

        Args:
            first: world cell index of cell (0, 0).
            ground: ground height per cell (nan = unknown).
            top: height of the highest thing above the ground per cell (nan = none).
            blocked: which cells are blocked.
        """
        self._first = np.asarray(first, dtype=np.int64)
        self._shape = ground.shape
        self._count = 0  # a build() after this places its points from scratch
        self.ground, self.top = ground, top
        self.closest = np.where(np.isfinite(ground), 0.0, np.inf)
        self.state = np.where(np.isfinite(ground), np.where(blocked, self.BLOCKED, self.FREE), self.UNKNOWN).astype(np.uint8)

    @property
    def origin(self) -> np.ndarray:
        """World (x, y) of cell (0, 0)'s corner."""
        return self._first * self.cell_size

    def cell_of(self, x: float, y: float) -> Cell:
        """Cell containing world point (x, y)."""
        i, j = self.cells_of([(x, y)])[0]
        return int(i), int(j)

    def cells_of(self, points) -> np.ndarray:
        """Cells (N, 2) containing world points (N, 2+); may lie outside the grid."""
        xy = np.asarray(points, dtype=float).reshape(len(points), -1)[:, :2]
        return (np.floor(xy / self.cell_size).astype(np.int64) - self._first).astype(int)

    def centers(self, cells) -> np.ndarray:
        """World (x, y) of cell centers."""
        return self.origin + (np.asarray(cells).reshape(-1, 2) + 0.5) * self.cell_size

    def inside(self, cells: np.ndarray) -> np.ndarray:
        """Which cells (N, 2) are on the grid."""
        return (cells[:, 0] >= 0) & (cells[:, 0] < self._shape[0]) & (cells[:, 1] >= 0) & (cells[:, 1] < self._shape[1])

    def state_at(self, x: float, y: float) -> int:
        """UNKNOWN, FREE or BLOCKED at world point (x, y)."""
        i, j = self.cell_of(x, y)
        return int(self.state[i, j]) if self.inside(np.array([[i, j]]))[0] else self.UNKNOWN

    def image(self, path_points: Optional[Iterable[Point]] = None,
              marks: Optional[Iterable[Point]] = None, pixels_per_cell: int = 3,
              structure_height: float = 1.1) -> np.ndarray:
        """The grid as an image

        Grey unknown, white free, black blocked by something structure_height or taller (walls, columns), brown
        other blocked cells; path_points in blue and marks in red.
        """
        colors = np.array([[150, 150, 150], [255, 255, 255], [20, 20, 20]], dtype=np.uint8)
        image = colors[self.state]                                   # (nx, ny, 3)
        with np.errstate(invalid="ignore"):
            low = (self.state == self.BLOCKED) & ~(self.top >= structure_height)
        image[low] = (150, 90, 40)
        for points, color in ((path_points, (30, 110, 255)), (marks, (230, 40, 40))):
            points = list(points or [])
            if points:
                cells = self.cells_of(points)
                cells = cells[self.inside(cells)]
                image[cells[:, 0], cells[:, 1]] = color
        image = np.flip(image.transpose(1, 0, 2), axis=0)             # rows = y, north at the top
        return np.repeat(np.repeat(image, pixels_per_cell, axis=0), pixels_per_cell, axis=1)
