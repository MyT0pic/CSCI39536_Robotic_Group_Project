"""IMU: orientation from the InertialUnit and turn rates from the Gyro."""
from robot.utils.types import Vector

class IMU:
    def __init__(self, robot, timestep: int):
        self.inertial_unit = robot.getDevice("inertial unit")
        self.gyro = robot.getDevice("gyro")
        for device in (self.inertial_unit, self.gyro):
            device.enable(timestep)

    def orientation(self) -> Vector:
        """(roll, pitch, yaw) in rad. Pitch is positive nose down."""
        return tuple(self.inertial_unit.getRollPitchYaw())

    def angular_velocity(self) -> Vector:
        """Rotation rate (x, y, z) in rad/s, about the robot's own axes."""
        return tuple(self.gyro.getValues())

