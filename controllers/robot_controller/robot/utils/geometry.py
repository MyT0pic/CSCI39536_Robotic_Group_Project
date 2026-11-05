import math
import numpy as np

def rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """3x3 rotation for roll about x, then pitch about y, then yaw about z."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])

def transform_points(points: np.ndarray, rotation: np.ndarray, translation) -> np.ndarray:
    """Rotate then move (N, 3) points."""
    return points @ rotation.T + np.asarray(translation, dtype=float)

def voxel_keys(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """One integer per voxel_size cube that each point falls in (cubes up to about 1 million apart per axis)."""
    cells = np.floor(points / voxel_size).astype(np.int64) + (1 << 20)
    return (cells[:, 0] << 42) | (cells[:, 1] << 21) | cells[:, 2]

def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Keep one point (the average) per voxel_size cube, to thin out a cloud."""
    if len(points) == 0:
        return points
    _, inverse, counts = np.unique(voxel_keys(points, voxel_size), return_inverse=True, return_counts=True)
    sums = np.column_stack([np.bincount(inverse, weights=points[:, k], minlength=len(counts)) for k in range(3)])
    return sums / counts[:, None]
