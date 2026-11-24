import datetime
import math
import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO
from controller import Supervisor, Keyboard


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
from robot.perception.victims import VictimApproach, VictimDetector

LOG_EVERY = 30
MAP_SAVE_EVERY = 120.0
FINISH_WAIT = 3.0  # s: once back home and done, wait this long, then pause the simulation
SETTLE_TIME = 1.0  # s
MAP_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "maps")
CONFIDENCE_THRESHOLD = 0.65
MANUAL_TIMEOUT = 2.0   # s without a key before autonomy takes back over
MANUAL_SPEED = 0.5     # m/s
MANUAL_TURN = 1.0      # rad/s
DETECT_EVERY = 10  # run YOLO every 10 steps (~0.16 s), it's too slow to run every step
LOG_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs")


class _Tee:
    """Print to the Webots console and to a log file at the same time."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


os.makedirs(LOG_FOLDER, exist_ok=True)
_log_file = open(os.path.join(LOG_FOLDER, datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S.txt")), "w", buffering=1)
sys.stdout = _Tee(sys.stdout, _log_file)

robot = Supervisor()
# physics step (16 ms)
timestep = int(robot.getBasicTimeStep())
dt = timestep / 1000.0

keyboard = robot.getKeyboard()
keyboard.enable(timestep)

def read_manual_command():
    """(speed m/s, turn rad/s) from arrow keys or WASD, or None if no drive key is held."""
    speed = turn = 0.0
    pressed = False
    key = keyboard.getKey()
    while key != -1:                       # several keys can be held at once
        k = key & Keyboard.KEY             # strip shift/ctrl/alt bits
        if k in (Keyboard.UP, ord("W")):
            speed += MANUAL_SPEED; pressed = True
        elif k in (Keyboard.DOWN, ord("S")):
            speed -= MANUAL_SPEED; pressed = True
        elif k in (Keyboard.LEFT, ord("A")):
            turn += MANUAL_TURN; pressed = True      # positive = left
        elif k in (Keyboard.RIGHT, ord("D")):
            turn -= MANUAL_TURN; pressed = True
        key = keyboard.getKey()
    return (speed, turn) if pressed else None

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
    if camera.hasRecognition():
        camera.recognitionEnable(timestep)

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
explorer.costmap.preferred_area = (-10.2, 10.2, -6.2, 6.2)  # the building's outer walls: drive outside only when there's no way inside

# victims: the camera's recognition finds them, the approach drives up to them
detector = VictimDetector()
approach = VictimApproach(detector)


def true_victim_positions():
    """Where the victims really are (supervisor), only to check the robot's estimates in the log."""
    children = robot.getRoot().getField("children")
    found = []
    for i in range(children.getCount()):
        node = children.getMFNode(i)
        if node.getTypeName() == "Pedestrian":
            x, y, _ = node.getPosition()
            found.append((x, y))
    return found


TRUE_VICTIMS = true_victim_positions()


def truth_error(victim):
    if not TRUE_VICTIMS:
        return ""
    error = min(math.dist((victim.x, victim.y), p) for p in TRUE_VICTIMS)
    return f" [{error:.2f} m from the real victim]"

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
       return False

   image_np = np.frombuffer(image, dtype=np.uint8).reshape((height, width, 4))
   image_BGR = cv2.cvtColor(image_np, cv2.COLOR_BGRA2BGR) # Convert from BGRA to BGR for YOLO processing
   model_results = model(image_BGR, iou=0.3, verbose=False)

   # Clear display
   clearing_image = display.imageNew(image, display.BGRA, width, height)
   display.imagePaste(clearing_image, 0, 0, False)
   display.imageDelete(clearing_image)

   found = False
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

           found = True 

           # Draw green bounding box
           display.setColor(0x00FF00)
           box_w = x2 - x1
           box_h = y2 - y1
           display.drawRectangle(x1, y1, box_w, box_h)

   return found

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
    manual_active = False
    last_manual = -1e9
    yolo_checks, yolo_hits = 0, 0  # YOLO checked while a victim was in view / YOLO saw a person
    was_approaching = False
    target_victim = None

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

        # VICTIMS: the camera's recognition, and YOLO alongside it (counted only while a victim is in view)
        if settled and camera:
            for victim in detector.update(camera, pose, robot.getTime()):
                print(f"[VICTIM] victim {victim.number} first seen at ({victim.x:.2f}, {victim.y:.2f}), "
                      f"t={robot.getTime():.1f} s{truth_error(victim)}")
        if camera and step_count % DETECT_EVERY == 0:
            yolo_saw = draw_victim_bounding_boxes()
            if detector.in_view:
                yolo_checks += 1
                yolo_hits += int(yolo_saw)

        # where to go next, as a forward speed (m/s) and turn rate (rad/s, positive = left)
        # manual override: any drive key takes control; no keys for MANUAL_TIMEOUT hands it back
        manual = read_manual_command()
        now = robot.getTime()
        if manual is not None:
            if not manual_active:
                manual_active = True
                forward_pid.reset()
                angular_pid.reset()
                print("[MODE] MANUAL")
            last_manual = now
        elif manual_active and now - last_manual > MANUAL_TIMEOUT:
            manual_active = False
            forward_pid.reset()
            angular_pid.reset()
            print("[MODE] AUTONOMOUS")

        target_speed, target_turn_rate = 0.0, 0.0
        command = None
        if not manual_active and settled and (approach.active or explorer.state in (explorer.EXPLORING, explorer.RETURNING)):
            command = approach.update(robot.getTime(), pose, explorer.costmap)
        if settled and not approach.active:
            if explorer.target_failed and target_victim is not None:
                target_victim.tries += 1
                target_victim.retry_after = robot.getTime() + (approach.retry_wait if target_victim.tries < 3 else math.inf)
            explorer.target_failed = False
            nearest = detector.nearest_unreported(pose, robot.getTime())
            target_victim = nearest[0] if nearest else None
            explorer.target = (target_victim.x, target_victim.y) if target_victim else None
        if manual_active:
            # keys held -> drive; keys just released -> hold still until the timeout expires
            target_speed, target_turn_rate = manual if manual is not None else (0.0, 0.0)
        elif command is not None:
            # driving to a victim: still through the local safety check, so it never drives into rubble
            target_speed, target_turn_rate = explorer.local.check(pose, *command)
        elif settled:
            stops = explorer.stops
            target_speed, target_turn_rate = explorer.update(robot.getTime(), pose, roll, pitch, slam.nav_points)
            if explorer.stops > stops:
                blocking = explorer.local.blocking
                nearest = blocking[int(np.argmin(np.hypot(blocking[:, 0] - pose.x, blocking[:, 1] - pose.y)))]
                print(f"[STOP] {robot.getTime():.1f} local cost map: {len(blocking)} cells in the way, nearest at "
                      f"({nearest[0]:.2f}, {nearest[1]:.2f})")
        if approach.active != was_approaching:
            forward_pid.reset()
            angular_pid.reset()
            if was_approaching:
                # back to exploring: the time spent with the victim isn't being stuck or making no progress
                now = robot.getTime()
                explorer._motion.clear()
                explorer._progress_time, explorer._best_left, explorer._next_plan = now, math.inf, now
            was_approaching = approach.active

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

            reported = sum(v.reported for v in detector.victims)
            print(f"  victims seen={len(detector.victims)} reported={reported} in_view={detector.in_view} "
                  f"yolo={yolo_hits}/{yolo_checks}")

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
            reported = sum(v.reported for v in detector.victims)
            print(f"[SUMMARY] t={finished_at:.1f} s, victims reported {reported} of 2, seen {len(detector.victims)}, "
                    f"YOLO agreed {yolo_hits}/{yolo_checks}, stops {explorer.stops}, tipping {explorer.tilt_events}")
            for v in detector.victims:
                    print(f"[SUMMARY] victim {v.number}: ({v.x:.2f}, {v.y:.2f}){truth_error(v)} "
                        f"{'reached' if v.reported else 'seen, not reached'}")


            
            if explorer.home is not None and math.dist((pose.x, pose.y), explorer.home) < 0.5:
                print(f"[DONE] Back home and finished exploring at {finished_at:.1f} s; pausing the simulation.")
            else:
                print(f"[DONE] Stopped at ({pose.x:.2f}, {pose.y:.2f}), couldn't get further, at {finished_at:.1f} s; "
                      f"pausing the simulation.")
            robot.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
            finished_at = None

        drive_with_pid(target_speed, target_turn_rate, measured_speed, measured_turn_rate)


if __name__ == "__main__":
    main()
