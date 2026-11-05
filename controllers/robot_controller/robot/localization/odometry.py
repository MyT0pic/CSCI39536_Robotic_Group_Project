"""Dead reckoning: pose (x, y, z, theta) from the wheel encoders, the gyro and the IMU's roll and pitch.

The gyro's turn rate gives the heading.
"""
import math
from typing import Optional, Sequence

from robot.localization.pose import Pose
from robot.utils.numeric import normalize_angle


class Odometry:
    def __init__(self, wheel_radius: float, distance_scale: float = 1.0):
        """Args:
            wheel_radius: m.
            distance_scale: correction factor for the wheel distance (wheel slip on the terrain).
        """
        self.wheel_radius = wheel_radius
        self.distance_scale = distance_scale

        self.pose = Pose()                           # set_pose places it in the world (else it starts at 0)
        self.speed = 0.0                             # m/s along the ground
        self.yaw_rate = 0.0                          # rad/s, how fast the heading changes
        self.distance = 0.0                          # total distance driven, m
        self._last_left: Optional[float] = None
        self._last_right: Optional[float] = None

    def set_pose(self, x: float, y: float, theta: float, z: Optional[float] = None) -> None:
        """Overwrite the pose, e.g. with SLAM's correction; z is kept if None."""
        self.pose.x, self.pose.y, self.pose.theta = x, y, normalize_angle(theta)
        if z is not None:
            self.pose.z = z

    def update(self, left_angle: float, right_angle: float, angular_velocity: Sequence[float],
               roll: float, pitch: float, dt: float) -> Pose:
        """Advance one step and return the pose.

        Args:
            left_angle, right_angle: wheel encoder angles (rad), each side's average.
            angular_velocity: gyro (x, y, z) in rad/s, about the robot's axes.
            roll, pitch: from the IMU (rad); used to split the wheel distance into ground distance and climb.
            dt: time since the last update (s).
        """
        if dt <= 0:
            raise ValueError("dt must be positive")
        pose = self.pose
        if self._last_left is None:  # first reading: nothing to compare against yet
            self._last_left, self._last_right = left_angle, right_angle
            return pose
        d_left = left_angle - self._last_left
        d_right = right_angle - self._last_right
        self._last_left, self._last_right = left_angle, right_angle

        # Heading change from the gyro, turned into a turn about the vertical so it
        # stays right when the robot is tilted on a slope
        _, wy, wz = angular_velocity
        self.yaw_rate = (math.sin(roll) * wy + math.cos(roll) * wz) / max(math.cos(pitch), 0.2)
        d_theta = self.yaw_rate * dt

        # Distance from the wheels, split into ground distance and height change
        distance = self.distance_scale * self.wheel_radius * (d_left + d_right) / 2
        ground = distance * math.cos(pitch)
        heading = pose.theta + d_theta / 2  # heading halfway through the step
        pose.x += ground * math.cos(heading)
        pose.y += ground * math.sin(heading)
        pose.z -= distance * math.sin(pitch)
        pose.theta = normalize_angle(pose.theta + d_theta)
        self.speed = ground / dt
        self.distance += abs(distance)

        return pose
