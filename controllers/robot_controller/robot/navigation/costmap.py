"""Cost of driving through each cell of a GridMap, and a collision check for commands.

The robot builds two: the global cost map from the SLAM grid (planning) and the local one from the local height
map (check() before the wheels move). Costs:
- infinite within robot_radius of a blocked cell;
- tight_cost extra within comfort_radius (the corners' turning circle), and extra fading out to safe_distance;
- unknown_cost extra for unknown cells (optimistic: exploring);
- cells the robot has driven count as free, and learned obstacles and marks as blocked.
"""
import math
from typing import Iterable

import numpy as np
from scipy import ndimage

from robot.localization.pose import Pose
from robot.mapping.grid_map import GridMap
from robot.utils.body import CORNER, HALF_LENGTH, HALF_WIDTH
from robot.utils.types import Cell, Command, Point


class CostMap:
    robot_radius = HALF_WIDTH + 0.05  # cells closer than this to an obstacle can't hold the robot's center
    comfort_radius = round(CORNER, 2)  # cells closer than this cost tight_cost extra (0.37)
    safe_distance = 0.7        # cells closer than this cost extra, up to near_wall_cost at comfort_radius
    unknown_cost = 0.5         # extra cost of an unknown cell (a free cell costs 1)
    tight_cost = 15.0
    near_wall_cost = 6.0
    visited_radius = 0.2       # cells this close to where the robot has driven count as free
    obstacle_radius = 0.15     # radius painted around a learned obstacle
    tight_trail_cost = 200.0   # cost of a driven cell closer than robot_radius to an obstacle
    # check()
    margin = 0.08              # room kept in front of the body ...
    side_margin = 0.03         # ... and beside it
    stop_time = 0.6            # look this far ahead (s) at the current speed
    stop_height = 0.15         # only obstacles at least this high stop the robot
    corner_radius = round(CORNER + 0.03, 2)  # turning in place with an obstacle this close ...
    slow_turn = 0.4            # ... is limited to this (rad/s)

    def __init__(self):
        self.blocked = False                 # check() stopped the last command ...
        self.blocking = np.empty((0, 2))     # ... for these cells (world x, y)

        self.grid = GridMap()
        self.state = np.zeros((0, 0), dtype=np.uint8)   # grid state with the extra layers applied
        self.clearance = np.zeros((0, 0))               # distance (m) to the nearest obstacle
        self.cost = np.zeros((0, 0))                    # cost per cell, inf where the robot can't go

    def build(self, grid: GridMap, visited: Iterable[Point] = (), obstacles: Iterable[Point] = (),
              marks: Iterable[Point] = ()) -> None:
        """Build the costs from a grid.

        Args:
            grid: the GridMap.
            visited: where the robot has driven (world x, y); the last one is where it stands, always free.
            obstacles: learned obstacles (world x, y), painted obstacle_radius wide.
            marks: obstacle cells (world x, y), blocking their own cell only.
        """
        self.grid = grid
        state = grid.state.copy()
        trail = np.zeros(state.shape, dtype=np.uint8)
        visited = list(visited)
        self._paint(trail, visited, self.visited_radius, 1)
        state[(trail == 1) & (state == GridMap.UNKNOWN)] = GridMap.FREE
        self._paint(state, visited[-1:], self.visited_radius, GridMap.FREE)
        self._paint(state, obstacles, self.obstacle_radius, GridMap.BLOCKED)
        self._paint(state, marks, 0.0, GridMap.BLOCKED)
        self.state = state

        blocked = state == GridMap.BLOCKED
        if blocked.any():
            self.clearance = ndimage.distance_transform_edt(~blocked) * grid.cell_size
        else:
            self.clearance = np.full(state.shape, np.inf)
        span = max(self.safe_distance - self.comfort_radius, 1e-6)
        closeness = np.clip((self.safe_distance - self.clearance) / span, 0.0, 1.0)
        cost = 1.0 + self.unknown_cost * (state == GridMap.UNKNOWN) + self.near_wall_cost * closeness
        cost[self.clearance < self.comfort_radius] += self.tight_cost
        cost[self.clearance < self.robot_radius] = np.inf
        tight = (trail == 1) & np.isinf(cost) & (state != GridMap.BLOCKED)
        cost[tight] = self.tight_trail_cost  # driven through before, so drivable, but avoided when possible
        self.cost = cost

    def free_around(self, cell: Cell, radius: float, cost: float = 50.0) -> None:
        """Give non-obstacle cells within radius (m) of cell a high finite cost, so the robot can drive out of a tight spot."""
        i, j = cell
        r = int(np.ceil(radius / self.grid.cell_size))
        lo_i, hi_i = max(i - r, 0), min(i + r + 1, self.cost.shape[0])
        lo_j, hi_j = max(j - r, 0), min(j + r + 1, self.cost.shape[1])
        window = self.cost[lo_i:hi_i, lo_j:hi_j]
        not_obstacle = self.state[lo_i:hi_i, lo_j:hi_j] != GridMap.BLOCKED
        window[np.isinf(window) & not_obstacle] = cost

    def check(self, pose: Pose, speed: float, turn: float) -> Command:
        """The command with forward motion stopped if it would hit an obstacle, and turning in place slowed near one.

        Checks the band the body sweeps along the commanded arc, out to stop_time at the current speed.
        """
        self.blocked = False
        self.blocking = np.empty((0, 2))
        if self.state.size == 0:
            return speed, turn
        with np.errstate(invalid="ignore"):
            cells = np.argwhere((self.state == GridMap.BLOCKED) & (self.grid.top >= self.stop_height))
        world = self.grid.centers(cells)
        c, s = math.cos(pose.theta), math.sin(pose.theta)
        dx, dy = world[:, 0] - pose.x, world[:, 1] - pose.y
        p = np.column_stack([c * dx + s * dy, -s * dx + c * dy])  # robot frame
        near = np.hypot(p[:, 0], p[:, 1]) < 2.0
        p, world = p[near], world[near]
        if len(p) == 0:
            return speed, turn
        half_cell = self.grid.cell_size / 2  # a cell's center is up to half a cell from what's in it
        if abs(speed) > 1e-3:
            reach = HALF_LENGTH + self.margin + abs(speed) * self.stop_time
            ahead, side = self._along_arc(p * [np.sign(speed), 1.0], turn / speed)  # backing up mirrors x, and the turn
            # Sideways no padding for the cell: the band must stay narrower than the planner's robot_radius (both
            # measure from cell centers), or every new path passing an obstacle at robot_radius is refused again
            in_path = ((ahead > HALF_LENGTH - 0.05) & (ahead < reach + half_cell) &
                       (np.abs(side) < HALF_WIDTH + self.side_margin))
            if in_path.any():
                self.blocked = True
                self.blocking = world[in_path]
                speed = 0.0
        if abs(speed) < 0.05 and abs(turn) > self.slow_turn:
            outside_body = (np.abs(p[:, 0]) > HALF_LENGTH) | (np.abs(p[:, 1]) > HALF_WIDTH)
            if np.any(outside_body & (np.hypot(p[:, 0], p[:, 1]) < self.corner_radius + half_cell)):
                turn = float(np.sign(turn)) * self.slow_turn
        return speed, turn

    @staticmethod
    def _along_arc(p: np.ndarray, curvature: float):
        """Points (robot frame, x along the motion) as (distance along the arc, offset from it) for this curvature (1/m)."""
        if abs(curvature) < 1e-3:
            return p[:, 0], p[:, 1]
        radius = 1.0 / abs(curvature)
        toward_center = p[:, 1] * np.sign(curvature)          # sideways, toward the center of the turn
        angle = np.arctan2(p[:, 0], radius - toward_center)   # how far round the turn
        return radius * angle, radius - np.hypot(p[:, 0], radius - toward_center)

    def passable(self, cell: Cell) -> bool:
        """Whether the robot's center can be at cell."""
        i, j = cell
        return 0 <= i < self.cost.shape[0] and 0 <= j < self.cost.shape[1] and bool(np.isfinite(self.cost[i, j]))

    def _paint(self, state: np.ndarray, points: Iterable[Point], radius: float, value: int) -> None:
        """Set value in every cell within radius (m) of the points."""
        points = np.asarray(list(points), dtype=float).reshape(-1, 2)
        if len(points) == 0 or state.size == 0:
            return
        cell = self.grid.cell_size
        r = int(np.ceil(radius / cell))
        offsets = np.array([(di, dj) for di in range(-r, r + 1) for dj in range(-r, r + 1)
                            if (di * di + dj * dj) * cell * cell <= radius * radius])
        cells = (self.grid.cells_of(points)[:, None, :] + offsets[None, :, :]).reshape(-1, 2)
        cells = cells[self.grid.inside(cells)]
        state[cells[:, 0], cells[:, 1]] = value
