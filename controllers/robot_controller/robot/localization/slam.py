"""Lidar-inertial SLAM: tracks the robot's pose and builds the 3D map.

Each sweep is leveled with the IMU, matched against a coarse surface map (scan_matching.py) starting from the
odometry guess, and accepted only if the match is good and close to the guess. Well-matched sweeps become
keyframes, added to the matching map and to the detailed point cloud. After several failed matches SLAM is
lost and searches mapped places near the last good pose.
"""
import math
from dataclasses import replace
from typing import List, Optional

import numpy as np

from robot.localization.pose import Pose
from robot.localization.scan_matching import Match, SurfaceMap
from robot.mapping.voxel_cloud import VoxelCloud
from robot.utils.geometry import rotation_matrix, transform_points, voxel_downsample
from robot.utils.numeric import normalize_angle


class LidarSLAM:
    # Settings (m and rad unless noted)
    scan_voxel = 0.3           # sweeps are thinned to one point per cube of this size before matching
    map_voxel = 0.2            # cube size of the matching map
    detail_voxel = 0.05        # cube size of the detailed map (closest look kept per cube)
    keyframe_distance = 0.5    # add a keyframe after moving this far ...
    keyframe_angle = math.radians(15)  # ... or turning this much
    match_distance = 1.0       # points farther than this from the map aren't matched
    iterations = 10            # ICP iterations per sweep
    normal_neighbors = 10      # neighbors used for each surface normal
    min_fitness = 0.25         # a good match has at least this share of points on a surface ...
    min_inliers = 150          # ... and this many ...
    max_rmse = 0.04            # ... at most this typical distance from it
    weak_direction = 0.01      # see SurfaceMap.match
    keyframe_fitness = 0.3     # least fitness for a sweep to become a keyframe
    max_correction = 0.25      # while tracking, the most a match may move the pose from odometry (+1 cm per failure)
    relocalize_fitness = 0.6   # least fitness for a pose found while lost ...
    relocalize_jump = 1.0      # ... and if it is farther than this from odometry ...
    relocalize_far_fitness = 0.8  # ... this much
    max_match_tilt = math.radians(30)  # sweeps aren't matched while tilted more ...
    max_match_turn_rate = 0.8  # ... or turning faster (rad/s): the gyro holds the heading better then
    max_keyframe_tilt = math.radians(20)  # no keyframes while tilted more ...
    max_keyframe_turn = math.radians(3)   # ... or after turning more since the last match (smeared sweep)
    lost_after = 3             # failed matches in a row before SLAM is lost
    search_after = 50          # while lost, sweeps matched from odometry before searching the map
    settle_sweeps = 10         # good sweeps needed after relocalizing before adding keyframes
    search_per_sweep = 12      # candidate poses tried per sweep while searching ...
    search_radius = 3.0        # ... within this distance of the last good pose

    def __init__(self):

        self.pose = Pose()
        self.keyframes: List[Pose] = []  # where each sweep added to the map was taken
        self.matched = False      # the last sweep matched the map well
        self.lost = False         # matching has failed several times in a row; searching for the pose
        self.fitness = 0.0        # share of the last sweep's points that matched
        self.rmse = 0.0           # typical distance (m) of matched points from their surfaces
        self.weak_directions = 0  # how many of (x, y, z, theta) the last sweep couldn't pin down
        self.relocalizations = 0  # times SLAM was lost and found its pose again

        self._map = SurfaceMap(self.map_voxel, self.normal_neighbors)  # coarse map for matching
        self._detail_map = VoxelCloud(self.detail_voxel)           # detailed 3D map: saved, and planned on
        self._last_good = Pose()
        self._failures = 0
        self._settling = 0
        self._candidates: List[np.ndarray] = []

    # ------------------------------------------------------------------ main step
    def update(self, scan: np.ndarray, roll: float, pitch: float, guess: Pose, extra: Optional[np.ndarray] = None,
               turn_rate: float = 0.0) -> Pose:
        """Match one sweep and return the corrected pose (odometry's guess if the match isn't trusted).

        Args:
            scan: (N, 3) main lidar points, robot frame.
            roll, pitch: IMU (rad).
            guess: odometry's pose.
            extra: more points of the same moment (near-field lidars) that go into the detailed map only.
            turn_rate: gyro turn rate (rad/s).
        """
        level = voxel_downsample(scan, self.scan_voxel) @ rotation_matrix(roll, pitch, 0.0).T
        too_tilted = max(abs(roll), abs(pitch)) > self.max_match_tilt
        spinning = abs(turn_rate) > self.max_match_turn_rate
        if len(level) < 100 or too_tilted or spinning:  # too little to go on, keep odometry's pose
            self.matched = False
            self.pose = replace(guess)
            return self.pose

        if not self.keyframes:  # first sweep starts the map
            self.pose = replace(guess)
            self._add_keyframe(scan, roll, pitch, extra)
            self.matched = True
            self._last_good = replace(self.pose)
            return self.pose

        # Lost only briefly: odometry is still good to a few cm, so keep matching from its guess. Only
        # after search_after failed sweeps in a row (e.g. the robot was carried off) search the map
        searching = self.lost and self._failures >= self.search_after
        if searching:
            match = self._search(level)
        else:
            match = self._map.match(level, self._as_vector(guess), self.iterations, self.match_distance,
                                    self.weak_direction)
            if not self._good(match):
                # The wheels may have spun without the robot moving: try the last good pose
                match = self._map.match(level, self._as_vector(self._last_good), self.iterations,
                                        self.match_distance, self.weak_direction)
            if not self._good(match) and self.lost:
                # A few degrees of heading error stops matching from converging in a long lane (far walls are
                # off by a metre or more): try the guess turned a little either way
                for offset in (2, -2, 4, -4, 6, -6):
                    start = self._as_vector(guess) + np.array([0.0, 0.0, 0.0, math.radians(offset)])
                    match = self._map.match(level, start, self.iterations, self.match_distance, self.weak_direction)
                    if self._good(match):
                        break

        self.fitness, self.rmse, self.weak_directions = match.fitness, match.rmse, match.weak_directions
        if searching:
            self.matched = self._good(match) and self._trust_relocalization(match, guess)
        else:
            self.matched = self._good(match) and self._small_correction(match, guess)
        if not self.matched:
            self._failures += 1
            if not self.lost and self._failures >= self.lost_after:
                self.lost = True
                self._candidates = self._search_candidates(guess.theta)
            self.pose = replace(guess)
            return self.pose

        if self.lost:
            self.relocalizations += 1
            self._settling = self.settle_sweeps  # don't add to the map until matching is steady again
        self.lost = False
        self._failures = 0
        e = match.estimate
        self.pose = Pose(x=e[0], y=e[1], theta=normalize_angle(e[3]), z=e[2])
        turned = abs(normalize_angle(self.pose.theta - self._last_good.theta))
        self._last_good = replace(self.pose)
        level_enough = (abs(roll) <= self.max_keyframe_tilt and abs(pitch) <= self.max_keyframe_tilt and
                        turned <= self.max_keyframe_turn)
        steady = self._settling == 0
        self._settling = max(0, self._settling - 1)
        if match.fitness >= self.keyframe_fitness and level_enough and steady and self._far_from_last_keyframe():
            self._add_keyframe(scan, roll, pitch, extra)
        return self.pose

    def _small_correction(self, match: Match, guess: Pose) -> bool:
        """Whether a tracking match stays within max_correction and 10 deg (more per failed sweep) of odometry."""
        moved = math.hypot(match.estimate[0] - guess.x, match.estimate[1] - guess.y)
        turned = abs(normalize_angle(match.estimate[3] - guess.theta))
        return (moved <= self.max_correction + 0.01 * self._failures and
                turned <= math.radians(10.0 + 0.5 * self._failures))

    def _trust_relocalization(self, match: Match, guess: Pose) -> bool:
        """Whether a pose found while lost is trustworthy: every direction constrained, and a better fit the farther it is from odometry."""
        if match.fitness < self.relocalize_fitness or match.weak_directions > 0:
            return False
        jump = math.hypot(match.estimate[0] - guess.x, match.estimate[1] - guess.y)
        return jump <= self.relocalize_jump or match.fitness >= self.relocalize_far_fitness

    def _good(self, match: Match) -> bool:
        return match.fitness >= self.min_fitness and match.inliers >= self.min_inliers and match.rmse <= self.max_rmse

    @staticmethod
    def _as_vector(pose: Pose) -> np.ndarray:
        return np.array([pose.x, pose.y, pose.z, pose.theta])

    # ------------------------------------------------------------------ relocalization
    def _search_candidates(self, heading: float) -> List[np.ndarray]:
        """Poses to try while lost: mapped places near the last good pose, each with several headings."""
        last = self._last_good
        spots = [(last.x, last.y, last.z)]
        for p in reversed(self.keyframes):
            if math.hypot(p.x - last.x, p.y - last.y) <= self.search_radius:
                spots.append((p.x, p.y, p.z))
        spots = list({(round(x * 2) / 2, round(y * 2) / 2): (x, y, z) for x, y, z in spots}.values())  # one per 0.5 m
        spots.sort(key=lambda s: math.hypot(s[0] - last.x, s[1] - last.y))
        headings = [heading + d for d in (0.0, 0.35, -0.35, 0.8, -0.8, 1.57, -1.57, math.pi)]
        return [np.array([x, y, z, h]) for x, y, z in spots for h in headings]

    def _search(self, level: np.ndarray) -> Match:
        """Try the next few candidate poses; return the best match."""
        if not self._candidates:
            self._candidates = self._search_candidates(self._last_good.theta)
        sample = level[np.random.default_rng(len(self._candidates)).choice(len(level), min(400, len(level)), replace=False)]
        best = Match(estimate=self._as_vector(self._last_good))
        for _ in range(min(self.search_per_sweep, len(self._candidates))):
            match = self._map.match(sample, self._candidates.pop(0), 6, 2.0 * self.match_distance, self.weak_direction)
            if match.fitness > best.fitness and match.rmse < 2 * self.max_rmse:
                best = match
        if best.hessian is not None:
            best = self._map.match(level, best.estimate, self.iterations, self.match_distance, self.weak_direction)
        return best

    # ------------------------------------------------------------------ map
    def _far_from_last_keyframe(self) -> bool:
        """Moved or turned enough since the last keyframe."""
        last = self.keyframes[-1]
        moved = math.hypot(self.pose.x - last.x, self.pose.y - last.y)
        turned = abs(normalize_angle(self.pose.theta - last.theta))
        return moved >= self.keyframe_distance or turned >= self.keyframe_angle

    def _add_keyframe(self, scan: np.ndarray, roll: float, pitch: float, extra: Optional[np.ndarray] = None) -> None:
        """Add the sweep (and the near-field points) to both maps at the current pose."""
        self.keyframes.append(replace(self.pose))
        points = voxel_downsample(scan, self.detail_voxel)
        extra = voxel_downsample(extra, self.detail_voxel) if extra is not None and len(extra) else np.empty((0, 3))
        rotation = rotation_matrix(roll, pitch, self.pose.theta)
        where = (self.pose.x, self.pose.y, self.pose.z)
        world = transform_points(points, rotation, where)
        ranges = np.hypot(points[:, 0], points[:, 1])  # how far from the robot (closer looks are more precise)
        self._map.add(world, ranges)
        self._detail_map.add(np.vstack([world, transform_points(extra, rotation, where)]),
                             np.concatenate([ranges, np.hypot(extra[:, 0], extra[:, 1])]))

    def map_points(self) -> np.ndarray:
        """The detailed map, (N, 3), one point per detail_voxel cube."""
        return self._detail_map.points

    def nav_points(self) -> np.ndarray:
        """The detailed map with the closest distance each point was seen from, (N, 4), for the grid map."""
        return np.column_stack([self._detail_map.points, self._detail_map.ranges])

    def save_map(self, path: str) -> int:
        """Write the detailed map as a binary .ply file; returns the number of points."""
        points = self.map_points().astype("<f4")
        with open(path, "wb") as f:
            f.write(b"ply\nformat binary_little_endian 1.0\n")
            f.write(f"element vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n".encode())
            f.write(points.tobytes())
        return len(points)
