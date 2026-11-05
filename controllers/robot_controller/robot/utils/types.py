from typing import Optional, Tuple

Point = Tuple[float, float]                  # world (x, y) in m
Cell = Tuple[int, int]                       # grid cell (i, j)
Vector = Tuple[float, float, float]          # (x, y, z)
Limits = Tuple[Optional[float], Optional[float]]  # (low, high); None means no limit
Command = Tuple[float, float]                # what to drive: (forward speed m/s, turn rate rad/s)
