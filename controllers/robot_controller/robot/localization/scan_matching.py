"""Point-to-plane matching of a lidar sweep against a map of surfaces."""
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from robot.mapping.voxel_cloud import VoxelCloud


@dataclass
class Match:
    """Result of matching one sweep."""
    estimate: np.ndarray           # (x, y, z, theta)
    fitness: float = 0.0           # share of the sweep's points lying on a map surface (inliers)
    rmse: float = float("inf")     # typical distance (m) of the inliers from their surfaces
    inliers: int = 0               # how many points lie on a map surface
    hessian: Optional[np.ndarray] = None
    weak_directions: int = 0       # how many of (x, y, z, theta) the scene couldn't pin down


class SurfaceMap:
    """Map points with a surface normal at each, in a k-d tree, for matching sweeps against."""

    def __init__(self, voxel_size: float, normal_neighbors: int = 10):
        self.cloud = VoxelCloud(voxel_size)
        self.normal_neighbors = normal_neighbors
        self.normals = np.empty((0, 3))
        self.tree = None

    def add(self, world_points: np.ndarray, ranges: Optional[np.ndarray] = None) -> None:
        """Add world points and update the normals around the new or replaced ones.

        Args:
            world_points: (N, 3).
            ranges: sensor distance of each point; a cube seen again from closer takes the closer point.
        """
        new = self.cloud.add(world_points, ranges)
        changed = np.concatenate([new, self.cloud.replaced])
        if len(changed) == 0:
            return
        cloud = self.cloud.points
        self.tree = cKDTree(cloud)
        self.normals = np.vstack([self.normals, np.zeros((len(new), 3))])
        k = min(self.normal_neighbors, len(cloud))
        _, neighbors = self.tree.query(cloud[changed], k=k)
        affected = np.unique(np.concatenate([changed, np.asarray(neighbors).reshape(-1)]))
        self.normals[affected] = self._surface_normals(affected)

    def _surface_normals(self, indices: np.ndarray) -> np.ndarray:
        """Surface normal at each map point, from the spread of its nearest neighbors."""
        cloud = self.cloud.points
        k = min(self.normal_neighbors, len(cloud))
        _, neighbors = self.tree.query(cloud[indices], k=k)
        near = cloud[np.asarray(neighbors).reshape(len(indices), k)]   # (N, k, 3)
        centered = near - near.mean(axis=1, keepdims=True)
        spread = np.einsum("nki,nkj->nij", centered, centered)          # (N, 3, 3)
        _, vectors = np.linalg.eigh(spread)
        return vectors[:, :, 0]                                         # least spread = surface normal

    def match(self, level: np.ndarray, start: np.ndarray, iterations: int, reach: float,
              weak_direction: float, inlier_distance: float = 0.1) -> Match:
        """Point-to-plane ICP for (x, y, z, theta), starting from start.

        Directions the scene doesn't constrain (fewer than weak_direction per point) keep the start's value.

        Args:
            level: the sweep, leveled with the IMU's roll and pitch (robot heading, not yet rotated).
            start: (x, y, z, theta) to start from.
            iterations: most ICP iterations.
            reach: points farther than this (m) from the map aren't matched (halved after 5 iterations).
            weak_direction: least constraint per matched point for a direction to be solved.
            inlier_distance: points this close (m) to a surface count as on it.
        """
        result = Match(estimate=np.array(start, dtype=float))
        if self.tree is None:
            return result
        residuals = np.empty(0)
        found_count = 0
        for iteration in range(iterations):
            x, y, z, theta = result.estimate
            c, s = math.cos(theta), math.sin(theta)
            world = level @ np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]).T + np.array([x, y, z])
            distance, index = self.tree.query(world, distance_upper_bound=reach if iteration < 5 else reach / 2)
            found = np.isfinite(distance)
            found_count = int(found.sum())
            if found_count < 30:
                return result
            normals = self.normals[index[found]]
            residuals = np.sum(normals * (world[found] - self.cloud.points[index[found]]), axis=1)

            # How each point moves when theta changes
            a = level[found]
            turn = np.stack([-s * a[:, 0] - c * a[:, 1], c * a[:, 0] - s * a[:, 1], np.zeros(len(a))], axis=1)
            jacobian = np.column_stack([normals, np.sum(normals * turn, axis=1)])
            weights = 1.0 / (1.0 + (residuals / 0.1) ** 2)  # robust: points far off their surface count less
            hessian = jacobian.T @ (jacobian * weights[:, None])
            gradient = jacobian.T @ (weights * residuals)

            # Only move in directions the scene actually pins down
            values, vectors = np.linalg.eigh(hessian)
            strong = values > weak_direction * found_count
            result.weak_directions = int((~strong).sum())
            result.hessian = hessian
            step = -vectors[:, strong] @ ((vectors[:, strong].T @ gradient) / values[strong])
            result.estimate = result.estimate + step
            if np.linalg.norm(step[:3]) < 1e-3 and abs(step[3]) < 1e-4:
                break
        inliers = np.abs(residuals) < inlier_distance
        result.inliers = int(inliers.sum())
        result.fitness = result.inliers / len(level)
        result.rmse = float(np.sqrt(np.mean(residuals[inliers] ** 2))) if inliers.any() else float("inf")
        return result
