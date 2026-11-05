from dataclasses import dataclass

@dataclass
class Pose:
    x: float = 0.0      # m
    y: float = 0.0      # m
    theta: float = 0.0  # rad, heading in [-pi, pi)
    z: float = 0.0      # m, height
