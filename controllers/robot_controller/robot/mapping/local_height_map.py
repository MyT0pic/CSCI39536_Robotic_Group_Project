"""Local height map: ground height and obstacles in a window around the robot, updated every lidar sweep.

Catches what the global grid can't: rocks and rubble a few cm high, and things that moved. Each sweep is
shifted onto the ground already known, folded into a per-cell ground
estimate, and compared with the median ground around each cell. A cell's obstacle score goes up when a sweep
shows something on it and down when it shows plain ground, so single odd sweeps don't count.
"""
import math
import numpy as np

from robot.localization.pose import Pose
from robot.mapping.grid_map import GridMap
from robot.utils.body import HALF_LENGTH, HALF_WIDTH
from robot.utils.numeric import nan_median, neighborhood


class LocalHeightMap:
    size = 8.0                 # window width, centered on the robot
    cell_size = GridMap.cell_size
    max_range = 4.0            # points farther than this are left out
    body = (HALF_LENGTH + 0.06, HALF_WIDTH + 0.05)  # points within this (half-length, half-width) are the robot
    max_height = 0.75          # points higher than this above the robot's center are ignored (ceilings)
    rise_height = 0.06         # a cell shows something when points stand this far above the ground around it ...
    min_points = 2             # ... at least this many of them
    window = 2                 # the ground around a cell is the median over this many cells each way
    mark_gain = 2              # score up per sweep showing something ...
    confirm = 4                # ... an obstacle from this score until it's back at 0 ...
    max_score = 8              # ... capped here; down 1 per sweep showing plain ground
    min_ground_points = 3      # points a cell needs in a sweep to count as seen plain
    align_cells = 30           # a sweep is shifted onto the known ground when they overlap in this many cells ...
    max_shift = 0.1            # ... by at most this much

    def __init__(self):
        self.n = int(round(self.size / self.cell_size))
        self.first = None                                   # world cell index of the window's (0, 0) cell
        self.ground = np.full((self.n, self.n), np.nan)     # ground height per cell (m, world z)
        self.weight = np.zeros((self.n, self.n))            # how much each ground estimate is worth
        self.rise = np.zeros((self.n, self.n))              # how far things stand above the ground (m), last seen
        self.score = np.zeros((self.n, self.n), dtype=np.int16)
        self.obstacle = np.zeros((self.n, self.n), dtype=bool)
        self.new = np.empty((0, 2))                          # world (x, y) of cells that became obstacles last update
        self.shift = 0.0                                     # how far (m) the last sweep was shifted onto the ground

    # ------------------------------------------------------------------ update
    def update(self, pose: Pose, points: np.ndarray) -> None:
        """Take a sweep: (N, 3) points in the robot frame, leveled with the IMU, at the robot's pose."""
        self._follow(pose)
        self.new = np.empty((0, 2))
        reach = np.hypot(points[:, 0], points[:, 1])
        body = (np.abs(points[:, 0]) < self.body[0]) & (np.abs(points[:, 1]) < self.body[1])
        keep = (reach < self.max_range) & ~body & (points[:, 2] < self.max_height)
        p, reach = points[keep], reach[keep]
        if len(p) == 0:
            return
        c, s = math.cos(pose.theta), math.sin(pose.theta)
        world = np.column_stack([pose.x + c * p[:, 0] - s * p[:, 1], pose.y + s * p[:, 0] + c * p[:, 1], pose.z + p[:, 2]])
        cells = np.floor(world[:, :2] / self.cell_size).astype(np.int64) - self.first
        inside = np.all((cells >= 0) & (cells < self.n), axis=1)
        world, reach, cells = world[inside], reach[inside], cells[inside]
        if len(world) == 0:
            return
        flat = cells[:, 0] * self.n + cells[:, 1]
        size = self.n * self.n
        z = world[:, 2]

        # This sweep's lowest point and point count per cell
        lowest = np.full(size, np.inf)
        np.minimum.at(lowest, flat, z)
        count = np.bincount(flat, minlength=size)
        nearest = np.full(size, np.inf)
        np.minimum.at(nearest, flat, reach)
        seen = count > 0

        # Shift the sweep onto the ground already known, where they overlap on plain ground
        known = seen & np.isfinite(self.ground.reshape(-1))
        self.shift = 0.0
        if known.sum() >= self.align_cells:
            gap = lowest[known] - self.ground.reshape(-1)[known]
            gap = gap[np.abs(gap) < self.max_shift]
            if len(gap) >= self.align_cells:
                self.shift = float(np.median(gap))
                z = z - self.shift
                lowest = lowest - self.shift

        # Fold the sweep's lowest points into the ground (a weighted average, close looks weighing more; the weight
        # is capped so the ground can still change)
        u = np.flatnonzero(seen)
        w = 1.0 / (0.5 + nearest[u])
        ground, weight = self.ground.reshape(-1), self.weight.reshape(-1)
        ground[u] = np.where(weight[u] > 0, (np.nan_to_num(ground[u]) * weight[u] + lowest[u] * w) / (weight[u] + w), lowest[u])
        weight[u] = np.minimum(weight[u] + w, 20.0)

        # Things standing above the ground around each cell
        around = nan_median(neighborhood(self.ground, self.window)).reshape(-1)
        rise = z - around[flat]
        with np.errstate(invalid="ignore"):
            up = np.bincount(flat[rise > self.rise_height], minlength=size)
            top = np.full(size, -np.inf)
            np.maximum.at(top, flat, np.nan_to_num(rise, nan=-np.inf))
        marked = up >= self.min_points
        plain = (count >= self.min_ground_points) & (up == 0) & np.isfinite(around)

        score = self.score.reshape(-1)
        score[marked] = np.minimum(score[marked] + self.mark_gain, self.max_score)
        score[plain] = np.maximum(score[plain] - 1, 0)
        self.rise.reshape(-1)[marked] = top[marked]
        was = self.obstacle.copy()
        obstacle = self.obstacle.reshape(-1)
        obstacle[score >= self.confirm] = True
        obstacle[score <= 0] = False
        fresh = self.obstacle & ~was
        if fresh.any():
            self.new = self.centers(np.argwhere(fresh))

    def _follow(self, pose: Pose) -> None:
        """Re-center the window on the robot (keeping the overlap) once it is 1 m off."""
        center = np.floor(np.array([pose.x, pose.y]) / self.cell_size).astype(np.int64) - self.n // 2
        if self.first is None:
            self.first = center
            return
        if np.max(np.abs(center - self.first)) * self.cell_size < 1.0:
            return
        shift = center - self.first
        for name, empty in (("ground", np.nan), ("weight", 0.0), ("rise", 0.0), ("score", 0), ("obstacle", False)):
            old = getattr(self, name)
            new = np.full_like(old, empty)
            src = tuple(slice(max(0, k), self.n + min(0, k)) for k in shift)
            dst = tuple(slice(max(0, -k), self.n - max(0, k)) for k in shift)
            new[dst] = old[src]
            setattr(self, name, new)
        self.first = center

    # ------------------------------------------------------------------ queries
    def centers(self, cells: np.ndarray) -> np.ndarray:
        """World (x, y) of window cells."""
        return (np.asarray(cells).reshape(-1, 2) + self.first + 0.5) * self.cell_size

    def obstacles(self, min_rise: float = 0.0, max_rise: float = np.inf) -> np.ndarray:
        """World (x, y) of obstacle cells standing between min_rise and max_rise (m) above the ground."""
        if self.first is None:
            return np.empty((0, 2))
        return self.centers(np.argwhere(self.obstacle & (self.rise >= min_rise) & (self.rise < max_rise)))

    def grid(self) -> GridMap:
        """The window as a GridMap, for building the local CostMap."""
        grid = GridMap()
        if self.first is not None:
            grid.set_layers(self.first, self.ground.copy(), np.where(self.obstacle, self.rise, np.nan), self.obstacle.copy())
        return grid

    def forget_near(self, x: float, y: float, radius: float) -> None:
        """Drop the obstacles within radius (m) of (x, y)."""
        if self.first is None:
            return
        centers = self.centers(np.indices((self.n, self.n)).reshape(2, -1).T).reshape(self.n, self.n, 2)
        near = np.hypot(centers[..., 0] - x, centers[..., 1] - y) <= radius
        self.obstacle[near] = False
        self.score[near] = 0
