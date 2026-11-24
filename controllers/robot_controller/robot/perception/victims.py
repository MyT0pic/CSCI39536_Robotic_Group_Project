"""Victims: finding them with the camera's Recognition node, and driving up to them.

The victims are Webots Pedestrian models (model "pedestrian"). The camera's Recognition node reports each one
it can see, with its position relative to the camera (x forward, y left, in metres). VictimDetector turns that
into a world position with the robot's SLAM pose and remembers each victim; VictimApproach drives to within a
safe standoff of the nearest one not yet reported, holds there, reports it and hands control back.
"""
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from robot.utils.numeric import clamp, normalize_angle

CAMERA_FORWARD = 0.22   # m: the camera sits this far forward of the robot centre (building_site.wbt)
SAME_VICTIM = 1.5       # m: sightings closer than this are the same victim
VICTIM_MODEL = "pedestrian"


@dataclass
class Victim:
    number: int
    x: float
    y: float
    first_seen: float
    sightings: int = 1
    reported: bool = False
    note: str = ""
    tries: int = 0
    retry_after: float = 0.0  # an approach that got stuck isn't tried again before this time


class VictimDetector:
    def __init__(self):
        self.victims: List[Victim] = []
        self.in_view = 0  # victims the camera sees right now

    def update(self, camera, pose, time: float) -> List[Victim]:
        """Read the camera's recognised objects; returns the victims seen for the first time."""
        new = []
        self.in_view = 0
        c, s = math.cos(pose.theta), math.sin(pose.theta)
        for obj in camera.getRecognitionObjects():
            if obj.getModel() != VICTIM_MODEL:
                continue
            self.in_view += 1
            position = obj.getPosition()
            fx, fy = position[0] + CAMERA_FORWARD, position[1]            # robot frame
            wx, wy = pose.x + c * fx - s * fy, pose.y + s * fx + c * fy  # world frame
            victim = self._match(wx, wy)
            if victim is None:
                victim = Victim(len(self.victims) + 1, wx, wy, time)
                self.victims.append(victim)
                new.append(victim)
            else:
                # running average, so the estimate settles as the robot sees it from more places
                victim.sightings += 1
                k = 1.0 / min(victim.sightings, 20)
                victim.x += k * (wx - victim.x)
                victim.y += k * (wy - victim.y)
        return new

    def nearest_unreported(self, pose, time: float = math.inf) -> Optional[Tuple[Victim, float, float]]:
        """(victim, distance m, bearing rad, + = left) of the closest victim not yet reported (and not waiting to be retried), or None."""
        best = None
        for v in self.victims:
            if v.reported or time < v.retry_after:
                continue
            dx, dy = v.x - pose.x, v.y - pose.y
            dist = math.hypot(dx, dy)
            if best is None or dist < best[1]:
                best = (v, dist, normalize_angle(math.atan2(dy, dx) - pose.theta))
        return best

    def _match(self, x: float, y: float) -> Optional[Victim]:
        for v in self.victims:
            if math.hypot(v.x - x, v.y - y) < SAME_VICTIM:
                return v
        return None


class VictimApproach:
    """Drive to standoff distance from the nearest unreported victim, hold, report, then let the explorer go on."""
    standoff = 1.2        # m, robot centre to victim centre (about 0.7 m of gap)
    start_within = 6.0    # m: only start an approach to a victim this close
    max_speed = 0.3       # m/s
    max_turn = 0.8        # rad/s
    hold_time = 3.0       # s stopped beside the victim before reporting
    give_up_after = 10.0  # s without getting 0.1 m closer: leave it to the explorer for now ...
    retry_wait = 20.0     # ... and try again after this long
    close_enough = 0.15   # m: stuck this close to the standoff distance counts as there

    def __init__(self, detector: VictimDetector):
        self.detector = detector
        self.target: Optional[Victim] = None
        self.holding_until: Optional[float] = None
        self._best = math.inf
        self._best_time = 0.0

    @property
    def active(self) -> bool:
        return self.target is not None

    def update(self, time: float, pose, costmap) -> Optional[Tuple[float, float]]:
        """(speed, turn) while approaching or holding; None when the explorer should drive."""
        if self.target is None:
            nearest = self.detector.nearest_unreported(pose, time)
            if nearest is None or nearest[1] > self.start_within or not self._clear_way(costmap, pose, nearest[0]):
                return None
            self.target, self._best, self._best_time = nearest[0], nearest[1], time
            print(f"[VICTIM] approaching victim {self.target.number} at ({self.target.x:.2f}, {self.target.y:.2f}),"
                  f" {nearest[1]:.2f} m away, t={time:.1f} s")
        v = self.target
        dist = math.hypot(v.x - pose.x, v.y - pose.y)
        bearing = normalize_angle(math.atan2(v.y - pose.y, v.x - pose.x) - pose.theta)

        if self.holding_until is not None:
            if time < self.holding_until:
                return 0.0, 0.0
            self._report(time, dist, "")
            return None
        stuck = time - self._best_time > self.give_up_after
        if dist <= self.standoff or (stuck and dist <= self.standoff + self.close_enough):
            self.holding_until = time + self.hold_time
            print(f"[VICTIM] stopped {dist:.2f} m from victim {v.number}, holding {self.hold_time:.0f} s")
            return 0.0, 0.0
        if dist < self._best - 0.1:
            self._best, self._best_time = dist, time
        elif stuck:
            v.retry_after = time + self.retry_wait
            print(f"[VICTIM] can't get closer to victim {v.number} from here ({dist:.2f} m), exploring on, t={time:.1f} s")
            self.target = None
            return None

        turn = clamp(1.5 * bearing, (-self.max_turn, self.max_turn))
        speed = 0.0 if abs(bearing) > 0.6 else clamp(0.5 * (dist - self.standoff), (0.0, self.max_speed))
        return speed, turn

    def _clear_way(self, costmap, pose, v: Victim) -> bool:
        """Whether the straight line from the robot to standoff distance from the victim is drivable on the cost map."""
        if costmap.cost.size == 0:
            return False
        dist = math.hypot(v.x - pose.x, v.y - pose.y)
        if dist <= self.standoff + self.close_enough:
            return True
        steps = int((dist - self.standoff) / 0.05) + 1
        for k in range(steps + 1):
            f = k * (dist - self.standoff) / steps / dist
            if not costmap.passable(costmap.grid.cell_of(pose.x + f * (v.x - pose.x), pose.y + f * (v.y - pose.y))):
                return False
        return True

    def _report(self, time: float, dist: float, note: str) -> None:
        v = self.target
        v.reported, v.note = True, note
        print(f"[VICTIM] REPORTED victim {v.number} at ({v.x:.2f}, {v.y:.2f}), robot {dist:.2f} m away, "
              f"t={time:.1f} s{note}")
        self.target, self.holding_until = None, None