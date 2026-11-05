"""The Pioneer 3-AT's size (m), shared by the controller, the maps and the cost maps."""
import math

HALF_LENGTH = 0.27   # center to the front and back of the body
HALF_WIDTH = 0.25    # center to the sides
CORNER = math.hypot(HALF_LENGTH, HALF_WIDTH)  # center to a corner: the circle it sweeps turning in place (0.37)
WHEEL_RADIUS = 0.11
TRACK_WIDTH = 0.394  # between the left and right wheels
