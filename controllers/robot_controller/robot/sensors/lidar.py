"""3D lidar: each sweep as a point cloud in the robot's frame."""
import math
import numpy as np

try:
    from controller.wb import wb 
except ImportError:
    wb = None

# One point of Webots' point cloud (WbLidarPoint): x, y, z, layer, time
_POINT = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("layer", "<i4"), ("time", "<f4")])


class Lidar:
    scans_per_second = 10.0
    edge_angle = math.radians(6)

    def __init__(self, robot, timestep: int, name: str = "lidar"):
        """Enable the lidar and its point cloud, and read where it is mounted on the robot.

        Args:
            robot: the Supervisor running this controller (gives the lidar's pose on the robot).
            timestep: controller step (ms).
            name: device name.
        """
        self.robot = robot
        self.device = robot.getDevice(name)
        self.period = max(timestep, int(round(1000 / self.scans_per_second / timestep)) * timestep)  # ms
        self.device.enable(self.period)
        self.device.enablePointCloud()
        pose = np.array(robot.getFromDevice(self.device._tag).getPose(robot.getSelf())).reshape(4, 4)
        self.rotation = pose[:3, :3]  # lidar frame -> robot frame
        self.mount = pose[:3, 3]      # where it sits on the robot (m)
        self.min_range = self.device.getMinRange()
        self.max_range = self.device.getMaxRange()
        self._last_scan_time = 0.0  # the first sweep is ready one period after the start

    def has_new_scan(self) -> bool:
        """True once per sweep period."""
        now = self.robot.getTime()
        if now - self._last_scan_time >= self.period / 1000 - 1e-6:
            self._last_scan_time = now
            return True
        return False

    def points(self) -> np.ndarray:
        """Latest sweep as (N, 3) points in the robot's frame, misses and out-of-range beams removed."""
        cloud = self._cloud()
        distance = np.linalg.norm(cloud, axis=1)
        valid = np.isfinite(distance) & (distance > self.min_range) & (distance < self.max_range)
        valid &= ~self._mixed_pixels(distance)
        return cloud[valid] @ self.rotation.T + self.mount

    def _mixed_pixels(self, distance: np.ndarray) -> np.ndarray:
        """Beams that grazed an edge (a wall end, a doorway side): the simulated lidar returns a distance between
        the edge and what's behind it, a point floating in the air. Two beams side by side whose distances jump
        almost along the beam mark such a spot; the farther one of the two is dropped."""
        layers, columns = self.device.getNumberOfLayers(), self.device.getHorizontalResolution()
        if layers * columns != len(distance) or columns < 2:
            return np.zeros(len(distance), dtype=bool)
        r = distance.reshape(layers, columns)
        step = self.device.getFov() / columns
        a, b = r[:, :-1], r[:, 1:]
        with np.errstate(invalid="ignore"):
            angle = np.arctan2(b * math.sin(step), a - b * math.cos(step))  # between beam a and the line to point b
            jump = (angle < self.edge_angle) | (angle > math.pi - self.edge_angle)
        mixed = np.zeros_like(r, dtype=bool)
        mixed[:, :-1] |= jump & (a > b)
        mixed[:, 1:] |= jump & (b >= a)
        return mixed.reshape(-1)

    def _cloud(self) -> np.ndarray:
        """Latest sweep as (N, 3) points in the lidar's frame (misses are inf), read from Webots' memory."""
        n = self.device.getNumberOfPoints()
        if wb is not None:
            raw = np.ctypeslib.as_array(wb.wb_lidar_get_point_cloud(self.device._tag), shape=(n * _POINT.itemsize,))
            raw = raw.view(_POINT)
            return np.column_stack([raw["x"], raw["y"], raw["z"]]).astype(float)
        return np.array([[p.x, p.y, p.z] for p in self.device.getPointCloud()], dtype=float).reshape(-1, 3)
