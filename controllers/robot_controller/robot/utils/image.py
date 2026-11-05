import cv2
import numpy as np


def save_png(path: str, image: np.ndarray) -> None:
    """Save an (height, width, 3) uint8 RGB image as a PNG file."""
    cv2.imwrite(path, cv2.cvtColor(np.ascontiguousarray(image, dtype=np.uint8), cv2.COLOR_RGB2BGR))


def save_top_view(path: str, points: np.ndarray, pixel: float = 0.03, top: float = 2.0) -> None:
    """Save a point cloud seen from above as a PNG: each pixel shows the highest point in it, colored by height
    (blue at the ground, through cyan, green and yellow, to red at `top` m above it); north (+y) is up."""
    if len(points) == 0:
        return
    height = points[:, 2] - np.percentile(points[:, 2], 2)
    low = points[:, :2].min(axis=0)
    cells = ((points[:, :2] - low) / pixel).astype(int)
    width, rows = cells[:, 0].max() + 1, cells[:, 1].max() + 1
    stops = np.array([[40, 60, 160], [40, 170, 220], [70, 190, 90], [240, 210, 60], [220, 60, 40]], dtype=float)
    t = np.clip(height / top, 0.0, 1.0) * (len(stops) - 1)
    i = np.minimum(t.astype(int), len(stops) - 2)
    f = (t - i)[:, None]
    colors = (stops[i] * (1 - f) + stops[i + 1] * f).astype(np.uint8)
    order = np.argsort(height)  # higher points drawn last, so they show
    image = np.full((rows, width, 3), 25, dtype=np.uint8)
    image[rows - 1 - cells[order, 1], cells[order, 0]] = colors[order]
    save_png(path, image)
