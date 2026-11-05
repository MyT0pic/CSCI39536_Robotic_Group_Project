import math
import numpy as np
from robot.utils.types import Limits


def clamp(value: float, limits: Limits) -> float:
    """Keep value within (low, high). None means no limit on that side."""
    low, high = limits
    if high is not None and value > high:
        return high
    if low is not None and value < low:
        return low
    return value

def normalize_angle(angle: float) -> float:
    """Wrap an angle in radians into [-pi, pi), so 179 and -179 degrees end up 2 degrees apart."""
    return (angle + math.pi) % (2 * math.pi) - math.pi

def neighborhood(values: np.ndarray, radius: int) -> np.ndarray:
    """Each cell's square window of (2 * radius + 1) ** 2 cells, stacked on a new first axis; nan past the edges."""
    nx, ny = values.shape
    padded = np.pad(values.astype(float), radius, constant_values=np.nan)
    span = range(-radius, radius + 1)
    return np.stack([padded[radius + di:radius + di + nx, radius + dj:radius + dj + ny] for di in span for dj in span])

def nan_median(stack: np.ndarray) -> np.ndarray:
    """Median over the first axis, skipping nans (nan where all are nan). Much faster than np.nanmedian
    for many short columns, and it raises no warnings for all-nan ones."""
    ordered = np.sort(stack, axis=0)  # nans sort last
    count = np.sum(~np.isnan(stack), axis=0)
    low = np.take_along_axis(ordered, np.maximum((count - 1) // 2, 0)[None], axis=0)[0]
    high = np.take_along_axis(ordered, np.maximum(count // 2, 0)[None], axis=0)[0]
    return np.where(count > 0, (low + high) / 2, np.nan)
