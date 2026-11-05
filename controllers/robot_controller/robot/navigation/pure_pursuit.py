"""Pure pursuit path following."""
import math
from typing import List, Optional, Sequence

from robot.localization.pose import Pose
from robot.utils.numeric import clamp, normalize_angle
from robot.utils.types import Command, Point


class PurePursuit:
    lookahead = 1.2            # how far ahead (m) along the path to aim
    max_speed = 0.65           # m/s
    min_speed = 0.1            # m/s
    max_turn_rate = 1.5        # rad/s
    turn_in_place_angle = math.radians(70)  # turn in place when the aim point is further off than this
    heading_gain = 2.0         # turn rate (rad/s) per rad of heading error when turning in place
    goal_tolerance = 0.25      # the path is done this close (m) to its end

    def __init__(self):
        self.path: List[Point] = []
        self._progress = 0  # index of the path point the robot has got to

    def set_path(self, points: Sequence[Point]) -> None:
        """Follow a new path (world x, y)."""
        self.path = [tuple(p) for p in points]
        self._progress = 0

    def remaining(self) -> List[Point]:
        """The part of the path still ahead."""
        return self.path[self._progress:]

    @property
    def done(self) -> bool:
        """Whether there is no path left."""
        return not self.path

    def command(self, pose: Pose, lookahead: Optional[float] = None) -> Command:
        """(speed m/s, turn rate rad/s) to follow the path from pose, aiming lookahead (m) ahead (default self.lookahead)."""
        lookahead = self.lookahead if lookahead is None else lookahead
        if not self.path:
            return 0.0, 0.0
        goal = self.path[-1]
        if math.hypot(goal[0] - pose.x, goal[1] - pose.y) < self.goal_tolerance:
            self.path = []
            return 0.0, 0.0

        # Move to the nearest path point in the next few metres
        window = self.path[self._progress:self._progress + 30]
        nearest = min(range(len(window)), key=lambda k: (window[k][0] - pose.x) ** 2 + (window[k][1] - pose.y) ** 2)
        self._progress += nearest

        aim = self._aim_point(pose, lookahead)
        dx, dy = aim[0] - pose.x, aim[1] - pose.y
        angle = normalize_angle(math.atan2(dy, dx) - pose.theta)
        limits = (-self.max_turn_rate, self.max_turn_rate)

        if abs(angle) > self.turn_in_place_angle:
            return 0.0, clamp(self.heading_gain * angle, limits)

        # Slower for sharper turns; arc through the aim point
        sharpness = abs(angle) / self.turn_in_place_angle
        speed = max(self.min_speed, self.max_speed * (1.0 - 0.7 * sharpness))
        distance = max(math.hypot(dx, dy), 1e-6)
        turn = clamp(speed * 2.0 * math.sin(angle) / distance, limits)
        return speed, turn

    def _aim_point(self, pose: Pose, lookahead: float) -> Point:
        """Where the path leaves the circle of radius lookahead around the robot (interpolated between path points), or its end."""
        for k in range(self._progress, len(self.path) - 1):
            (x0, y0), (x1, y1) = self.path[k], self.path[k + 1]
            if math.hypot(x1 - pose.x, y1 - pose.y) < lookahead:
                continue
            # first t in [0, 1] with |p0 + t (p1 - p0) - robot| = lookahead
            dx, dy, fx, fy = x1 - x0, y1 - y0, x0 - pose.x, y0 - pose.y
            a = dx * dx + dy * dy
            b = 2 * (fx * dx + fy * dy)
            c = fx * fx + fy * fy - lookahead * lookahead
            root = math.sqrt(max(b * b - 4 * a * c, 0.0))
            t = (-b + root) / (2 * a) if a > 1e-12 else 1.0
            t = min(max(t, 0.0), 1.0)
            return x0 + t * dx, y0 + t * dy
        return self.path[-1]
