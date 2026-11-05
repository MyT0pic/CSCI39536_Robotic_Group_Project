"""Frontier exploration: map the whole reachable site, then drive home.

Each plan (every replan_every s, or sooner when needed) rebuilds the global cost map, picks a frontier (the
current one while it's still worth it, else the cheapest by path cost plus turn penalty) and sets the path
to it. Each lidar sweep updates the local height map and local cost map, whose obstacles go into the global
one and against which every command is checked. When stuck, tipping or repeatedly stopped, the robot marks
the spot, backs up and replans. With no frontier left it returns home (RETURNING) and stops (DONE).
"""
import math
from collections import Counter, deque
from typing import Callable, List, Optional, Tuple
import numpy as np
from scipy import ndimage
from robot.localization.pose import Pose
from robot.mapping.grid_map import GridMap
from robot.mapping.local_height_map import LocalHeightMap
from robot.navigation.costmap import CostMap
from robot.navigation.frontiers import find_frontiers, unexplored
from robot.navigation.pure_pursuit import PurePursuit
from robot.navigation.planner import PathSearch, along, corners, line_cells
from robot.utils.types import Cell, Command, Point



class Explorer:
    EXPLORING, BACKING_UP, RETURNING, DONE = "EXPLORING", "BACKING_UP", "RETURNING", "DONE"

    # Settings (m and s unless noted)
    replan_every = 1.0         # time between plans
    min_frontier_size = 8      # smallest frontier (cells)
    min_goal_distance = 0.8    # frontiers closer than this are ignored
    blind_radius = 1.0         # unknown cells this close to the trail count as explored (ground under the robot)
    explore_range = 2.5        # floor counts as explored once seen from this close (not through a doorway)
    goal_reach = 0.5           # a frontier is reached from a roomy cell within this distance ...
    goal_beyond = 2.0          # ... and the path goes on up to this far past it over drivable ground
    min_travel = 0.3           # a frontier still unexplored with a shorter path is given up on for now
    min_reachable = 200        # fewer reachable cells than this is a planning glitch ...
    boxed_time = 3.0           # ... and if it lasts this long, the robot's own marks within 1 m are forgotten
    max_attempts = 3           # tries per goal; given-up goals are retried only when no fresh frontier is left
    gain_radius = 2.0          # a cell's gain is the unexplored area (m2) within this distance
    min_gain = 0.5             # a frontier needs at least min_gain ...
    gain_per_m = 0.1           # ... + gain_per_m per metre of path (m2) of gain
    progress_distance = 0.3    # the path left must shrink by this much ...
    progress_timeout = 12.0    # ... this often, or the goal is given up on
    max_back_ups = 2           # back-offs on the way to one goal before giving it up
    turn_penalty = 1.5         # extra cost (m) per rad of turning toward a frontier
    goal_keep_radius = 1.0     # the current goal stays while its frontier is this close
    keep_path_slack = 1.1      # keep the current path unless a new one is cheaper by more than this factor
    stuck_time = 3.0           # driving forward this long ...
    stuck_distance = 0.1       # ... without moving this far is stuck
    max_tilt = math.radians(18)  # tilt at which the robot backs off
    back_up_time = 1.5
    back_up_speed = 0.2        # m/s
    done_after = 3.0           # no reachable frontier for this long ends exploring
    home_tolerance = 0.3       # home when this close to the start
    slow_factor = 0.4          # share of full speed right next to obstacles ...
    full_speed_clearance = 0.5  # ... rising to full speed with this much room
    min_lookahead = 0.5        # pursuit lookahead next to obstacles, rising to its own in the open

    def __init__(self):
        self.grid = GridMap()            # global grid, from the SLAM map
        self.costmap = CostMap()         # global cost map: planning
        self.pursuit = PurePursuit()
        self.height_map = LocalHeightMap()
        self.local = CostMap()           # local cost map, from the height map: the check before the wheels

        self.state = self.EXPLORING
        self.goal: Optional[Point] = None     # frontier being driven to (world x, y)
        self.frontier_count = 0
        self.visited: List[Point] = []        # where the robot has been, every 0.1 m
        self.obstacles: List[Point] = []      # learned obstacles: where it got stuck or started to tip over
        self.given_up: List[List[float]] = []  # [x, y, failed tries] of goals the robot gave up on
        self.give_ups: Counter = Counter()      # how often it gave up on a goal, by reason
        self.stuck_events = 0
        self.tilt_events = 0
        self.back_up_reason = ""
        self.stops = 0                # times the local cost map's check stopped the robot
        self.unboxed = 0              # times the robot's own marks boxed it in and the nearest were forgotten
        self._boxed_since = None
        self._stops: List[Tuple[float, float, float]] = []  # (time, x, y) of recent stops
        self._next_stop = 0.0

        self._next_plan = 0.0
        self._last_plan = -math.inf
        self._best_left = math.inf   # shortest path left to the goal seen so far (m) ...
        self._progress_time = 0.0    # ... and when it last shrank by progress_distance
        self._goal_back_ups = 0      # back-offs on the way to the current goal
        self._back_until = 0.0
        self._no_goal_since = None
        self._speed = 0.0
        self._motion = deque()  # (time, x, y, commanded speed)
        self._resume = self.EXPLORING  # what to go back to after backing up
        self.home: Optional[Point] = None  # where the robot started; it returns here when done

    def update(self, time: float, pose: Pose, roll: float, pitch: float,
               map_points: Callable[[], np.ndarray]) -> Command:
        """Advance one step; returns (forward speed m/s, turn rate rad/s).

        Args:
            time: sim time (s).
            pose: robot pose.
            roll, pitch: IMU (rad), for the tip-over check.
            map_points: returns SLAM's map (N, 4); called only when planning.
        """
        if self.home is None:
            self.home = (pose.x, pose.y)
        self._remember(pose)
        self._motion.append((time, pose.x, pose.y, self._speed))
        while self._motion and self._motion[0][0] < time - self.stuck_time:
            self._motion.popleft()

        if self.state == self.BACKING_UP:
            if time < self._back_until:
                return self._safe(time, pose, -self.back_up_speed, 0.0)
            self.state = self._resume
            self._next_plan = time
            self._progress_time = time  # backing up isn't lack of progress

        if self.state == self.RETURNING and math.hypot(pose.x - self.home[0], pose.y - self.home[1]) < self.home_tolerance:
            self.state = self.DONE
            self.goal = None
            self.pursuit.set_path([])

        if self.state in (self.EXPLORING, self.RETURNING):
            if max(abs(roll), abs(pitch)) > self.max_tilt and self._speed > 0:
                self.tilt_events += 1
                return self._back_up(time, pose, "tilting")
            if self._stuck(time):
                self.stuck_events += 1
                return self._back_up(time, pose, "stuck")
            if self._held(time, pose):
                # Replanning after each stop didn't help: the planner lets the robot drive out of the cells
                # right around it, so up close it keeps choosing the way the check refuses
                return self._back_up(time, pose, "blocked")
            if self.state == self.EXPLORING and self.goal is not None and self._no_progress(time):
                self._give_up(self.goal, "no progress")
                self._next_plan = time

        # Replan on schedule, or early when the path is used up (at most 5 times a second: a plan takes
        # up to about 0.1 s, so replanning every step while there is no path would stall the robot)
        path_done = self.state in (self.EXPLORING, self.RETURNING) and self.pursuit.done and time >= self._last_plan + 0.2
        if self.state != self.DONE and (time >= self._next_plan or path_done):
            self._plan(time, pose, map_points)
        if self.state == self.DONE:
            return self._drive(0.0, 0.0)
        # In the open: full speed, and aim far along the path for a straight line. Next to obstacles: slow,
        # and aim close, so the robot follows the path round corners and doorways instead of cutting them
        room = self._room(pose)
        lookahead = self.min_lookahead + room * (self.pursuit.lookahead - self.min_lookahead)
        speed, turn = self.pursuit.command(pose, lookahead)
        return self._safe(time, pose, speed * max(room, self.slow_factor), turn)

    @property
    def path(self) -> List[Point]:
        """The path being followed (world x, y)."""
        return self.pursuit.path

    def add_sweep(self, time: float, pose: Pose, points: np.ndarray) -> None:
        """Update the local maps with a sweep (robot frame, leveled, (N, 3)); plan now if a new obstacle is on the path."""
        self.height_map.update(pose, points)
        self.local.build(self.height_map.grid())
        new = self.height_map.new
        rest = self.pursuit.remaining()
        if len(new) and len(rest):
            path = np.asarray(rest)
            gap = np.min(np.hypot(new[:, None, 0] - path[None, :, 0], new[:, None, 1] - path[None, :, 1]))
            if gap < self.costmap.robot_radius + 0.1:
                self._next_plan = min(self._next_plan, time)

    def _safe(self, time: float, pose: Pose, speed: float, turn: float) -> Command:
        """Check the command against the local cost map; on a stop, plan again now and count the stop."""
        speed, turn = self.local.check(pose, speed, turn)
        if self.local.blocked and time >= self._next_stop:
            self._next_stop = time + 0.5
            self.stops += 1
            self._stops.append((time, pose.x, pose.y))
            self._motion.clear()          # standing still for the check isn't being stuck
            self._next_plan = time        # replan at once
            self._last_plan = -math.inf
        return self._drive(speed, turn)

    # ------------------------------------------------------------------ planning
    def _plan(self, time: float, pose: Pose, map_points: Callable[[], np.ndarray]) -> None:
        """Rebuild the maps, choose a frontier and set the path; head home when none is left."""
        self._next_plan = time + self.replan_every
        self._last_plan = time
        searched = self._search_from(time, pose, map_points)
        if searched is None:
            return
        start, search = searched
        state, gain = self._exploration_state()
        # Frontier cells are often too close to a wall for the robot's center, so a frontier counts
        # as reached from the cheapest reachable cell within goal_reach of it
        reach = int(math.ceil(self.goal_reach / self.grid.cell_size))
        # ... and only from a cell with room to turn around in: the robot will have to turn there for its next
        # goal, and in a pocket barely wider than itself (e.g. between a desk and a wall) its corners wedge it in.
        # Paths still go through tight gaps; they just don't end in one
        to_roomy = np.where(self.costmap.clearance >= self.costmap.comfort_radius, search.distance, np.inf)
        near_cost = ndimage.minimum_filter(to_roomy, size=2 * reach + 1, mode="constant", cval=np.inf)
        candidates = self._frontier_goals(pose, state, gain, near_cost)
        goals, none_worth_it = self._ordered_goals(candidates, gain, near_cost)
        for cell, goal in goals:
            if self._go_to(time, pose, cell, goal, to_roomy, reach, start, search):
                return

        if self._no_goal_since is None:
            self._no_goal_since = time
        self.goal = None
        self.pursuit.set_path([])
        if self.state == self.RETURNING or none_worth_it or time - self._no_goal_since >= self.done_after:
            # Everything reachable is explored: head home (and keep looking on the way: if a frontier worth it
            # shows up again, explore it first)
            self.state = self.RETURNING
            self._plan_home(search)

    def _search_from(self, time: float, pose: Pose, map_points: Callable[[], np.ndarray]) -> Optional[Tuple[Cell, PathSearch]]:
        """Build the grid and global cost map and search from the robot: (start cell, search), or None if there's nothing to plan on."""
        points = map_points()
        if len(points) == 0:
            return None
        self.grid.build(points)
        # The local map's obstacles: low ones (rocks) as learned obstacles, which mark a little around them (only a
        # rock's top stands out enough to be marked, its sides don't); tall ones (walls, drums) cell by cell, so
        # walls don't grow and doorways stay as wide as they are. Where the robot stands is drivable whatever the
        # map says (a slightly wrong pose or a stray point must not wall it in)
        low = self.height_map.obstacles(max_rise=self.local.stop_height)
        tall = self.height_map.obstacles(min_rise=self.local.stop_height)
        self.costmap.build(self.grid, self.visited + [(pose.x, pose.y)], self.obstacles + [tuple(p) for p in low], tall)
        start = self.grid.cell_of(pose.x, pose.y)
        if not self.grid.inside(np.array([start]))[0]:
            return None
        # Next to an obstacle (e.g. right after backing off from it) every cell around the robot may be too
        # close to plan through: let it drive out, or it would wait there for good
        self.costmap.free_around(start, self.costmap.robot_radius + 0.1)
        search = PathSearch(self.costmap.cost, start)
        if np.isfinite(search.distance).sum() < self.min_reachable:
            # Nowhere to go from here: usually a planning glitch (e.g. the pose briefly wrong), not the end of
            # exploring, so keep the current path and try again next time. If it lasts, the robot's own marks
            # (back-offs, the local map's rocks) have boxed it in: forget the ones right around it. Walls and
            # anything else the lidar map shows stay, and real rubble is found again within a few sweeps
            if self._boxed_since is None:
                self._boxed_since = time
            elif time - self._boxed_since >= self.boxed_time:
                self._forget_obstacles_near(pose, 1.0)
                self._boxed_since = None
                self._next_plan = time
            return None
        self._boxed_since = None
        return start, search

    def _exploration_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Cell states for exploring (far-only ground counts as unexplored), and each cell's gain (m2)."""
        state = self.costmap.state.copy()
        # Ground seen only from far away (through a doorway, across a room) is mapped too thinly for a detailed
        # 3D map: few points, at a grazing angle, nothing behind the furniture. It counts as unexplored until
        # the lidar has seen it up close, so the robot goes into each room instead of looking in from the door
        if self.grid.closest.shape == state.shape:
            state[(state == GridMap.FREE) & ~(self.grid.closest <= self.explore_range)] = GridMap.UNKNOWN
        # The lidar can't see the ground close around itself, so unknown cells near where the robot has
        # driven would stay unknown forever; they're as explored as they'll get
        state[(state == GridMap.UNKNOWN) & self._near_visited(self.blind_radius)] = GridMap.FREE
        n = 2 * int(round(self.gain_radius / self.grid.cell_size)) + 1
        gain = ndimage.uniform_filter(unexplored(state).astype(float), size=n, mode="constant") * (n * self.grid.cell_size) ** 2
        return state, gain

    def _frontier_goals(self, pose: Pose, state: np.ndarray, gain: np.ndarray, near_cost: np.ndarray) -> List[dict]:
        """One candidate per frontier: cell, world point, path cost (m), gain, turn (rad), and whether it is the current goal."""
        frontiers = find_frontiers(state, self.min_frontier_size)
        self.frontier_count = len(frontiers)
        candidates = []
        for frontier in frontiers:
            cells = frontier.cells
            world = self.grid.centers(cells)
            cost = near_cost[cells[:, 0], cells[:, 1]] * self.grid.cell_size
            ok = np.isfinite(cost) & (np.hypot(world[:, 0] - pose.x, world[:, 1] - pose.y) >= self.min_goal_distance)
            # Goals given up on are skipped while fresh frontiers are left; then they're tried again (from
            # somewhere else they may be reachable after all) until max_attempts. Going back and forth to retry
            # them while new ground is still waiting wastes the most time of all
            fresh, retry = ok.copy(), ok.copy()
            for x, y, tries in self.given_up:
                near = np.hypot(world[:, 0] - x, world[:, 1] - y) <= self.goal_keep_radius
                fresh &= ~near
                if tries >= self.max_attempts:
                    retry &= ~near
            ok = fresh if fresh.any() else retry
            if not ok.any():
                continue
            # Aim at the middle of the frontier, not its nearest point: as the robot drives up to it the
            # near edge keeps moving back, so a goal there jumps every replan and the robot keeps turning
            middle = world[ok].mean(axis=0)
            k = int(np.argmin(np.where(ok, np.hypot(world[:, 0] - middle[0], world[:, 1] - middle[1]), np.inf)))
            area = float(gain[cells[k, 0], cells[k, 1]])
            if area < self.min_gain:
                continue  # a sliver of unknown: not worth the trip
            turn = abs(math.remainder(math.atan2(world[k, 1] - pose.y, world[k, 0] - pose.x) - pose.theta, math.tau))
            # The frontier being driven to: it shrinks and moves back as the robot sees more of it.
            # Sticking with it until it's explored, unreachable or given up on keeps the robot from
            # changing its mind (and turning) as the map fills in
            current = (self.goal is not None and
                       np.min(np.hypot(world[:, 0] - self.goal[0], world[:, 1] - self.goal[1])) < self.goal_keep_radius)
            candidates.append({"cell": tuple(cells[k]), "goal": (float(world[k, 0]), float(world[k, 1])),
                               "cost": float(cost[k]), "gain": area, "turn": turn, "current": current,
                               "retry": not fresh.any()})
        if any(not c["retry"] for c in candidates):
            candidates = [c for c in candidates if not c["retry"]]
        return candidates

    def _ordered_goals(self, candidates: List[dict], gain: np.ndarray,
                       near_cost: np.ndarray) -> Tuple[List[Tuple[Cell, Point]], bool]:
        """Goals worth the trip, best first, as (cell, world point); and whether there were frontiers but none worth it."""
        # A frontier is worth it only if it reveals enough for the trip; the current one stays (its gain
        # shrinks as the robot gets there)
        worth = [c for c in candidates if c["current"] or c["gain"] >= self.min_gain + self.gain_per_m * c["cost"]]
        # The current frontier first, then the cheapest to get to: path cost plus a penalty for turning toward it,
        # so the robot finishes what's ahead before turning back. (A route through all the frontiers, planned like
        # a travelling salesman's, did no better in tests: the frontiers change as the robot drives, so the route
        # is outdated a few seconds after it's planned)
        worth.sort(key=lambda c: (not c["current"], c["cost"] + self.turn_penalty * c["turn"]))
        goals = [(c["cell"], c["goal"]) for c in worth]
        # Stick with the goal the robot is driving to while there is still unexplored ground around it and it
        # can be reached. The frontier near it moves as the lidar uncovers ground (and pieces get re-cut), so
        # matching the goal to today's frontiers loses it every few seconds and the robot turns to a new one
        committed = self._committed_goal(gain, near_cost)
        if committed is not None:
            goals = [committed] + [g for g in goals if math.dist(g[1], committed[1]) > self.goal_keep_radius]
        return goals, bool(candidates) and not worth

    def _go_to(self, time: float, pose: Pose, cell: Cell, goal: Point, to_roomy: np.ndarray, reach: int,
               start: Cell, search: PathSearch) -> bool:
        """Set the path to a frontier goal; False if the robot is already as close as it can get."""
        i, j = cell
        # Drive to the cheapest reachable cell near the frontier
        window = to_roomy[max(i - reach, 0):i + reach + 1, max(j - reach, 0):j + reach + 1]
        di, dj = np.unravel_index(int(np.argmin(window)), window.shape)
        target = (max(i - reach, 0) + di, max(j - reach, 0) + dj)
        if math.dist(self.grid.centers([target])[0], (pose.x, pose.y)) < max(self.min_travel, self.pursuit.goal_tolerance + 0.1):
            # Already as close as the robot can get from here, yet still unexplored (e.g. hidden
            # behind rubble): leave it for now; it may be visible from another side later
            self._give_up(goal, "can't get closer")
            return False
        cells, path = self._route(search.path_to(self._past(target, start, search)))
        self._no_goal_since = None
        self.state = self.EXPLORING
        same_goal = self.goal is not None and math.dist(goal, self.goal) < self.goal_keep_radius
        if same_goal and self._current_path_still_good(cells):
            # Keep following the path the robot is on: switching to a slightly different one every
            # second makes it weave, especially on bumpy ground where the map keeps changing a little
            self.goal = goal
            return True
        if self.goal is None or math.dist(goal, self.goal) > 0.5:
            self._best_left, self._progress_time = math.inf, time  # new goal: start the progress watch
            self._goal_back_ups = 0
        self.goal = goal
        self.pursuit.set_path(path.tolist())
        return True

    def _committed_goal(self, gain: np.ndarray, near_cost: np.ndarray) -> Optional[Tuple[Cell, Point]]:
        """The current goal as (cell, world point) if it still has gain, is reachable and isn't given up on."""
        if self.state != self.EXPLORING or self.goal is None:
            return None
        cell = self.grid.cells_of([self.goal])
        if not self.grid.inside(cell)[0]:
            return None
        i, j = int(cell[0, 0]), int(cell[0, 1])
        given_up = any(math.dist(self.goal, g[:2]) <= self.goal_keep_radius for g in self.given_up)
        if given_up or gain[i, j] < self.min_gain or not np.isfinite(near_cost[i, j]):
            return None
        return (i, j), self.goal

    def _current_path_still_good(self, new_cells: np.ndarray) -> bool:
        """Whether the rest of the current path is passable and at most keep_path_slack times new_cells' cost."""
        rest = self.pursuit.remaining()
        if len(rest) < 2 or len(new_cells) < 2:
            return False
        old = self.grid.cells_of(rest)
        if not self.grid.inside(old).all():
            return False
        old_cost = self.costmap.cost[old[:, 0], old[:, 1]]
        if not np.all(np.isfinite(old_cost)):
            return False
        def total(cells, costs):
            return float(np.sum(np.hypot(*np.diff(cells, axis=0).T) * costs[1:]))
        new_cost = self.costmap.cost[new_cells[:, 0], new_cells[:, 1]]
        return total(old, old_cost) <= self.keep_path_slack * total(new_cells, new_cost)

    def _plan_home(self, search: PathSearch) -> None:
        """Path home, or to the reachable cell closest to it; DONE if already there."""
        reachable = np.argwhere(np.isfinite(search.distance))
        if len(reachable) == 0:
            self.pursuit.set_path([])
            return
        i, j = self.grid.cell_of(*self.home)
        gap = np.hypot(reachable[:, 0] - i, reachable[:, 1] - j)
        # Closest to home first; among equally close cells, the cheapest to get to
        k = int(np.argmin(gap * 1e3 + search.distance[reachable[:, 0], reachable[:, 1]]))
        cells, path = self._route(search.path_to(tuple(reachable[k])))
        if len(cells) <= 2:  # already as close to home as it can get
            self.state = self.DONE
            self.pursuit.set_path([])
            return
        self.goal = self.home
        self.pursuit.set_path(path.tolist())

    def _past(self, target: Cell, start: Cell, search: PathSearch) -> Cell:
        """Extend a goal cell up to goal_beyond (m) further from the robot over drivable cells, so the robot drives through the frontier."""
        di, dj = target[0] - start[0], target[1] - start[1]
        norm = math.hypot(di, dj)
        if norm < 1e-6:
            return target
        best = target
        for k in range(1, int(self.goal_beyond / self.grid.cell_size) + 1):
            cell = (int(round(target[0] + di / norm * k)), int(round(target[1] + dj / norm * k)))
            inside = 0 <= cell[0] < search.shape[0] and 0 <= cell[1] < search.shape[1]
            if not inside or not search.reachable(cell) or self.costmap.clearance[cell] < self.costmap.comfort_radius:
                break
            best = cell
        return best

    def _route(self, raw: List[Cell]) -> Tuple[np.ndarray, np.ndarray]:
        """Straighten a grid path: the cells it crosses, and world points every 10 cm to drive."""
        ends = corners(raw, self.costmap.cost)
        return np.array(line_cells(ends)), along(self.grid.centers(ends), 0.1)

    # ------------------------------------------------------------------ trouble
    def _no_progress(self, time: float) -> bool:
        """Whether the path left hasn't shrunk by progress_distance for progress_timeout."""
        rest = self.pursuit.remaining()
        if len(rest) < 2:
            return False
        left = float(np.sum(np.hypot(*np.diff(np.array(rest), axis=0).T)))
        if left < self._best_left - self.progress_distance:
            self._best_left, self._progress_time = left, time
            return False
        return time - self._progress_time > self.progress_timeout

    def _stuck(self, time: float) -> bool:
        """Whether the robot drove forward for stuck_time without moving stuck_distance."""
        if not self._motion or self._motion[0][0] > time - self.stuck_time + 0.1:
            return False
        if any(speed <= 0.05 for _, _, _, speed in self._motion):
            return False
        _, x0, y0, _ = self._motion[0]
        _, x1, y1, _ = self._motion[-1]
        return math.hypot(x1 - x0, y1 - y0) < self.stuck_distance

    def _learn_obstacles(self, pose: Pose, points: np.ndarray) -> None:
        """Add points (robot frame, (N, 2)) as learned obstacles, one per grid cell."""
        c, s = math.cos(pose.theta), math.sin(pose.theta)
        world = np.column_stack([pose.x + c * points[:, 0] - s * points[:, 1], pose.y + s * points[:, 0] + c * points[:, 1]])
        size = self.grid.cell_size
        for x, y in np.unique((np.floor(world / size) + 0.5) * size, axis=0).tolist():
            if (x, y) not in self.obstacles:
                self.obstacles.append((x, y))

    def _forget_obstacles_near(self, pose: Pose, radius: float) -> None:
        """Drop learned and local obstacles within radius (m) of the robot."""
        self.obstacles = [p for p in self.obstacles if math.hypot(p[0] - pose.x, p[1] - pose.y) > radius]
        self.height_map.forget_near(pose.x, pose.y, radius)
        self.unboxed += 1

    def _held(self, time: float, pose: Pose) -> bool:
        """Whether the check stopped the robot 3 times in 6 s without it moving."""
        recent = [(t, x, y) for t, x, y in self._stops if t > time - 6.0]
        self._stops = recent
        if len(recent) < 3 or any(math.hypot(x - pose.x, y - pose.y) > 0.15 for _, x, y in recent):
            return False
        self._stops = []
        return True

    def _back_up(self, time: float, pose: Pose, reason: str = "") -> Command:
        """Mark the front of the robot as an obstacle and back up; give up the goal after max_back_ups."""
        # Mark the whole front of the robot, not just straight ahead: what stopped or tipped it is often under
        # one front wheel (a low rock the lidar can't tell from the bumpy ground), and a single mark in the
        # middle lets the next path slide past it and put the same wheel back on it
        self._learn_obstacles(pose, np.array([[0.4, 0.0], [0.3, 0.2], [0.3, -0.2]]))
        self.back_up_reason = reason
        self._goal_back_ups += 1
        if self.goal is not None and self._goal_back_ups >= self.max_back_ups:
            self._give_up(self.goal, "backed off")
        self._resume = self.state
        self.state = self.BACKING_UP
        self._back_until = time + self.back_up_time
        self._motion.clear()
        self.pursuit.set_path([])
        return self._drive(-self.back_up_speed, 0.0)

    def _near_visited(self, radius: float) -> np.ndarray:
        """Cells within radius (m) of the robot's trail."""
        shape = self.grid.state.shape
        cells = self.grid.cells_of(self.visited) if self.visited else np.empty((0, 2), dtype=int)
        cells = cells[self.grid.inside(cells)]
        if len(cells) == 0:
            return np.zeros(shape, dtype=bool)
        away = np.ones(shape, dtype=bool)
        away[cells[:, 0], cells[:, 1]] = False
        return ndimage.distance_transform_edt(away) * self.grid.cell_size <= radius

    def _give_up(self, goal: Point, reason: str) -> None:
        """Count a failed try at goal."""
        self.give_ups[reason] += 1
        entry = next((g for g in self.given_up if math.hypot(g[0] - goal[0], g[1] - goal[1]) < 0.5), None)
        if entry is None:
            entry = [goal[0], goal[1], 0]
            self.given_up.append(entry)
        entry[2] += 1

    def _remember(self, pose: Pose) -> None:
        """Add the pose to the trail (every 10 cm)."""
        if not self.visited or math.hypot(pose.x - self.visited[-1][0], pose.y - self.visited[-1][1]) >= 0.1:
            self.visited.append((pose.x, pose.y))

    def _room(self, pose: Pose) -> float:
        """Room around the robot: 0 at robot_radius from an obstacle, 1 at full_speed_clearance or more."""
        if self.costmap.clearance.size == 0:
            return 1.0
        cell = self.grid.cells_of([(pose.x, pose.y)])
        if not self.grid.inside(cell)[0]:
            return 1.0
        tight, open_ = self.costmap.robot_radius, self.full_speed_clearance
        f = (self.costmap.clearance[cell[0, 0], cell[0, 1]] - tight) / max(open_ - tight, 1e-6)
        return float(np.clip(f, 0.0, 1.0))

    def _drive(self, speed: float, turn: float) -> Command:
        """Return the command, remembering the speed."""
        self._speed = speed
        return speed, turn
