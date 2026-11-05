import os

import cv2
import numpy as np
from ultralytics import YOLO
from controller import Supervisor

from robot.control.pid import PIDController
from robot.localization.odometry import Odometry
from robot.localization.slam import LidarSLAM
from robot.navigation.explorer import Explorer
from robot.sensors.imu import IMU
from robot.sensors.lidar import Lidar
from robot.utils.body import TRACK_WIDTH, WHEEL_RADIUS
from robot.utils.geometry import rotation_matrix
from robot.utils.image import save_png, save_top_view
from robot.utils.numeric import clamp

LOG_EVERY = 30
MAP_SAVE_EVERY = 120.0
FINISH_WAIT = 3.0  # s: once back home and done, wait this long, then pause the simulation
SETTLE_TIME = 1.0  # s
MAP_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "maps")
CONFIDENCE_THRESHOLD = 0.65


robot = Supervisor()
# physics step (16 ms)
timestep = int(robot.getBasicTimeStep())
dt = timestep / 1000.0

def get_wheels(names):
    """The wheel motors, in velocity mode, stopped."""
    wheels = []
    for name in names:
        wheel = robot.getDevice(name)
        wheel.setPosition(float("inf"))
        wheel.setVelocity(0.0)
        wheels.append(wheel)
    return wheels


def get_encoders(names):
    """The wheel encoders, enabled."""
    encoders = []
    for name in names:
        encoder = robot.getDevice(name)
        encoder.enable(timestep)
        encoders.append(encoder)
    return encoders


left_wheels = get_wheels(["front left wheel", "back left wheel"])
right_wheels = get_wheels(["front right wheel", "back right wheel"])
left_encoders = get_encoders(["front left wheel sensor", "back left wheel sensor"])
right_encoders = get_encoders(["front right wheel sensor", "back right wheel sensor"])
MOTOR_LIMIT = left_wheels[0].getMaxVelocity()  # rad/s, the wheel motors' top speed


#Turn on the front camera
camera = robot.getDevice("camera")
if camera:
	camera.enable(timestep)

# Pre-trained YOLO Model for Object Detection and Classification
model = YOLO("yolov8n.pt")

# COCO dataset class index for a person is 0
PERSON_CLASS_ID = 0

# To force Display View to overlay Camera View
robot.step(timestep)

display = robot.getDevice("display")
display.attachCamera(camera)

# odom
imu = IMU(robot, timestep)
odometry = Odometry(wheel_radius=WHEEL_RADIUS, distance_scale=0.95)

# slam
lidar = Lidar(robot, timestep)
# near-field lidars, tilted down, for the ground and low obstacles close around the robot
front_lidar = Lidar(robot, timestep, "front lidar")
rear_lidar = Lidar(robot, timestep, "rear lidar")
slam = LidarSLAM()

# navigation: explore until the whole reachable area is mapped
explorer = Explorer()

forward_pid = PIDController(kp=0.3, ki=3.0, output_limits=(-MOTOR_LIMIT, MOTOR_LIMIT))   # m/s in, rad/s out
angular_pid = PIDController(kp=1.0, ki=36.0, output_limits=(-MOTOR_LIMIT, MOTOR_LIMIT))  # rad/s in, rad/s out
SKID_TURN_FACTOR = 1.0


def drive(left, right):
    """Set the left and right wheel velocities (rad/s)."""
    for wheel in left_wheels:
        wheel.setVelocity(left)
    for wheel in right_wheels:
        wheel.setVelocity(right)


def average_position(encoders):
    """Mean angle (rad) of the encoders."""
    return sum(encoder.getValue() for encoder in encoders) / len(encoders)


def drive_with_pid(target_speed, target_turn_rate, measured_speed, measured_turn_rate):
    """Run the speed and turn-rate PIDs and set the wheel velocities."""
    forward = forward_pid.update(measured_speed, dt, setpoint=target_speed,
                                 feedforward=target_speed / WHEEL_RADIUS)
    turn = angular_pid.update(measured_turn_rate, dt, setpoint=target_turn_rate,
                              feedforward=SKID_TURN_FACTOR * target_turn_rate * TRACK_WIDTH / (2 * WHEEL_RADIUS))
    limits = (-MOTOR_LIMIT, MOTOR_LIMIT)
    drive(clamp(forward - turn, limits), clamp(forward + turn, limits))


def near_field_points():
    """The near-field lidars' latest sweeps together, robot frame."""
    return np.vstack([front_lidar.points(), rear_lidar.points()])


def save_maps(final=False):
    """Save slam_map.ply and slam_grid.png to maps/, and slam_map_top.png when final."""
    os.makedirs(MAP_FOLDER, exist_ok=True)
    slam.save_map(os.path.join(MAP_FOLDER, "slam_map.ply"))
    marks = list(explorer.obstacles) + [tuple(p) for p in explorer.height_map.obstacles()]
    save_png(os.path.join(MAP_FOLDER, "slam_grid.png"), explorer.grid.image(path_points=explorer.visited, marks=marks))
    if final:
        save_top_view(os.path.join(MAP_FOLDER, "slam_map_top.png"), slam.map_points())
    print(f"[SLAM] Saved maps: {len(slam.map_points())} points, {len(slam.keyframes)} keyframes"
          + (f" -> {os.path.abspath(MAP_FOLDER)}" if final else ""))


def find_orange_target():
    """Look for the orange target in the camera image; returns (found, offset from center -1..1)."""
    if not camera:
        return False, 0.0
    width = camera.getWidth()
    height = camera.getHeight()
    image = camera.getImage()

    #Target RGB color thresholds (CURRENTLY GONNA BE LIMITED TO A ORANGE SEARCH TARGET SIGNATURE WILL CHANGE SOON)
    #ORANGE
    TARGET_R, TARGET_G, TARGET_B = 255, 100, 0
    TOLERANCE = 50

    #Scan the image frame rapidly to look for
    for x in range(0, width, 5):
        for y in range(0, height, 5):
            r = camera.imageGetRed(image, width, x, y)
            g = camera.imageGetGreen(image, width, x, y)
            b = camera.imageGetBlue(image, width, x, y)
            if abs(r - TARGET_R) < TOLERANCE and abs(g - TARGET_G) < TOLERANCE and abs(b - TARGET_B) < TOLERANCE:
                return True, (x - (width / 2)) / (width / 2)
    return False, 0.0

def draw_victim_bounding_boxes():
   width = camera.getWidth()
   height = camera.getHeight()
   image = camera.getImage()

   if not image:
       return

   image_np = np.frombuffer(image, dtype=np.uint8).reshape((height, width, 4))
   image_BGR = cv2.cvtColor(image_np, cv2.COLOR_BGRA2BGR) # Convert from BGRA to BGR for YOLO processing
   model_results = model(image_BGR, iou=0.3, verbose=False)

   # Clear display
   clearing_image = display.imageNew(image, display.BGRA, width, height)
   display.imagePaste(clearing_image, 0, 0, False)
   display.imageDelete(clearing_image)

   for result in model_results:
       boxes = result.boxes
       highest_confidence = 0.0

       # Stores class_id, class_name, confidence level, coordinates in the form [x1, y1, x2, y2]
       highest_confidence_box = {"class_id": -1, "class_name": "", "confidence": 0.0, "coords": [None, None, None, None]}

       for box in boxes:
           class_id = int(box.cls[0])
           class_name = result.names[class_id]
           confidence = float(box.conf[0])
           x1, y1, x2, y2 = map(int, box.xyxy[0]) # Get bounding box coordinates

           if class_id == PERSON_CLASS_ID and confidence > highest_confidence:
               highest_confidence = confidence
               highest_confidence_box["class_id"] = class_id
               highest_confidence_box["class_name"] = class_name
               highest_confidence_box["confidence"] = confidence
               highest_confidence_box["coords"] = [x1, y1, x2, y2]

       if highest_confidence_box["class_id"] == PERSON_CLASS_ID and highest_confidence_box["confidence"] > CONFIDENCE_THRESHOLD:
           confidence = highest_confidence_box["confidence"]
           bounding_box = highest_confidence_box["coords"]
           x1, y1, x2, y2 = bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]

           print(f"Human detected with highest confidence: {confidence} with bounding box: [{x1}, {y1}, {x2}, {y2}]")

           # Draw green bounding box
           display.setColor(0x00FF00)
           box_w = x2 - x1
           box_h = y2 - y1
           display.drawRectangle(x1, y1, box_w, box_h)

def start_pose():
    """The robot's start pose in the world: position from the supervisor, heading from the IMU."""
    x, y, z = robot.getSelf().getPosition()
    return x, y, imu.orientation()[2], z


def main():
    step_count = 0
    placed = False  # odometry set to the start pose (once the robot has landed)
    previous_state = explorer.state
    next_map_save = MAP_SAVE_EVERY
    finished_at = None  # when the robot got back home with nothing left to explore

    print("[SUCCESS] First Responder robot_controller is connected and streaming code!")

    while robot.step(timestep) != -1:
        # Where the robot is
        roll, pitch, _ = imu.orientation()
        pose = odometry.update(average_position(left_encoders), average_position(right_encoders),
                               imu.angular_velocity(), roll, pitch, dt)
        measured_speed = odometry.speed
        measured_turn_rate = odometry.yaw_rate

        # SLAM: match each lidar sweep to the map
        settled = robot.getTime() >= SETTLE_TIME
        if settled and not placed:
            odometry.set_pose(*start_pose())
            placed = True
        if settled and lidar.has_new_scan():
            main_points, near_points = lidar.points(), near_field_points()
            corrected = slam.update(main_points, roll, pitch, guess=pose, extra=near_points,
                                    turn_rate=odometry.yaw_rate)
            if slam.matched:
                odometry.set_pose(corrected.x, corrected.y, corrected.theta, corrected.z)
            # the local height map and cost map: every sweep, all lidars, leveled, at the corrected pose
            explorer.add_sweep(robot.getTime(), pose,
                               np.vstack([main_points, near_points]) @ rotation_matrix(roll, pitch, 0.0).T)
        if robot.getTime() >= next_map_save:
            save_maps()
            next_map_save += MAP_SAVE_EVERY

        # VISION DETECTION (off for now: the camera isn't used for navigation yet)
        # target_found, center_offset = find_orange_target()

        #where to go next, as a forward speed (m/s) and turn rate (rad/s, positive = left)
        target_speed, target_turn_rate = 0.0, 0.0
        if settled:
            stops = explorer.stops
            target_speed, target_turn_rate = explorer.update(robot.getTime(), pose, roll, pitch, slam.nav_points)
            if explorer.stops > stops:
                blocking = explorer.local.blocking
                nearest = blocking[int(np.argmin(np.hypot(blocking[:, 0] - pose.x, blocking[:, 1] - pose.y)))]
                print(f"[STOP] {robot.getTime():.1f} local cost map: {len(blocking)} cells in the way, nearest at "
                      f"({nearest[0]:.2f}, {nearest[1]:.2f})")

        step_count += 1
        if step_count % LOG_EVERY == 0:
            goal = f"({explorer.goal[0]:.1f}, {explorer.goal[1]:.1f})" if explorer.goal else "none"
            gave_up = ", ".join(f"{reason} {n}" for reason, n in explorer.give_ups.items()) or "0"
            print(f"{explorer.state} goal={goal} frontiers={explorer.frontier_count} "
                  f"stuck={explorer.stuck_events} tipping={explorer.tilt_events} gave_up=({gave_up}) "
                  f"spots={len(explorer.given_up)} unboxed={explorer.unboxed} stops={explorer.stops} "
                  f"local_obstacles={len(explorer.height_map.obstacles())}")
                  # f"target_seen={target_found}")
            print(f"  pose x={pose.x:.2f} y={pose.y:.2f} z={pose.z:.2f} heading={pose.theta:.2f} "
                  f"speed={odometry.speed:.2f} turn={odometry.yaw_rate:.2f}")
            print(f"  slam {'LOST' if slam.lost else 'matched' if slam.matched else 'NOT matched'} "
                  f"fit={slam.fitness:.2f} rmse={slam.rmse:.3f} keyframes={len(slam.keyframes)} "
                  f"relocalized={slam.relocalizations}")

        if explorer.state != previous_state:
            forward_pid.reset()
            angular_pid.reset()
            reason = f" ({explorer.back_up_reason})" if explorer.state == explorer.BACKING_UP else ""
            print(f"[NAV] {previous_state} -> {explorer.state}{reason}")
            if explorer.state == explorer.DONE:
                save_maps(final=True)  # back home and done exploring: the finished maps
                finished_at = robot.getTime()
            previous_state = explorer.state

        if finished_at is not None and robot.getTime() >= finished_at + FINISH_WAIT:
            print(f"[DONE] Back home and finished exploring at {finished_at:.1f} s; pausing the simulation.")
            robot.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
            finished_at = None

        drive_with_pid(target_speed, target_turn_rate, measured_speed, measured_turn_rate)


if __name__ == "__main__":
    main()
