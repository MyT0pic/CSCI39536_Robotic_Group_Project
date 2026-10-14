import sys
from controller import Robot

robot = Robot()
timestep = int(robot.getBasicTimeStep())

print("[SUCCESS] First Responder robot_controller is connected and streaming code!")

while robot.step(timestep) != -1:
    # Your search, track, and delivery algorithms will live inside this loop
    pass

