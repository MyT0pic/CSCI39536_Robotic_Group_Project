from controller import Robot

MAX_SPEED = 5.0          
OBSTACLE_THRESHOLD = 850.0  
CLOSE_LEVEL = 850.0         
HOLD_STEPS = 60             
BACK_STEPS = 25             
TURN_STEPS = 40             
LOG_EVERY = 30              

FORWARD, BACK, TURN,TRACKING = "FORWARD", "BACK", "TURN", "TRACKING"

robot = Robot()
timestep = int(robot.getBasicTimeStep())


def get_wheels(names):
    wheels = []
    for name in names:
        wheel = robot.getDevice(name)
        wheel.setPosition(float("inf"))
        wheel.setVelocity(0.0)
        wheels.append(wheel)
    return wheels


left_wheels = get_wheels(["front left wheel", "back left wheel"])
right_wheels = get_wheels(["front right wheel", "back right wheel"])

# so0-so7 are the front-facing sonars
sonars = [robot.getDevice(f"so{i}") for i in range(8)]
for sonar in sonars:
    sonar.enable(timestep)


#Turn on the front camera
camera = robot.getDevice("camera")
if camera:
	camera.enable(timestep)



def drive(left, right):
    for wheel in left_wheels:
        wheel.setVelocity(left)
    for wheel in right_wheels:
        wheel.setVelocity(right)


def main():
    state = FORWARD
    back_count = 0
    turn_count = 0
    hold = 0
    turn_left = False
    step_count = 0

    print("[SUCCESS] First Responder robot_controller is connected and streaming code!")

    while robot.step(timestep) != -1:
        values = [sonar.getValue() for sonar in sonars]
        front = max(values[1:7])
        left = max(values[0:3])
        right = max(values[5:8])

        # VISION DETECTION
        target_found = False
        center_offset = 0.0

        if camera:
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
                    r = camera.imageGetRed(image,width, x, y)
                    g = camera.imageGetGreen(image, width, x, y)
                    b = camera.imageGetBlue(image,width, x, y)
                    
                    if abs(r - TARGET_R) < TOLERANCE and abs(g - TARGET_G) < TOLERANCE and abs(b - TARGET_B) < TOLERANCE:
                        target_found = True
                        center_offset = (x -(width / 2)) / (width / 2)
                        break
                    if target_found:
                        break



        step_count += 1
        if step_count % LOG_EVERY == 0:
            print("sonar:", [round(v) for v in values], state)


        if front > CLOSE_LEVEL:
            hold = HOLD_STEPS
        elif hold > 0:
            hold -= 1
        blocked = front > OBSTACLE_THRESHOLD or hold > 0


        #Target Locked Tracking Prioity Override 
        if target_found and state in [FORWARD, TRACKING]:
            state = TRACKING
            l_speed = MAX_SPEED * (0.5 + center_offset * 0.4)
            r_speed = MAX_SPEED * (0.5 - center_offset * 0.4)
            drive(l_speed, r_speed)

        elif state == FORWARD:
            if blocked:
                state = BACK
                back_count = BACK_STEPS
                turn_left = left < right  # turn toward the clearer side
                hold = 0
            else:
                drive(MAX_SPEED, MAX_SPEED)
        elif state == BACK:
            drive(-MAX_SPEED, -MAX_SPEED)
            back_count -= 1
            if back_count <= 0:
                state = TURN
                turn_count = TURN_STEPS
        elif state == TURN:
            if turn_left:
                drive(-MAX_SPEED, MAX_SPEED)
            else:
                drive(MAX_SPEED, -MAX_SPEED)
            turn_count -= 1
            if turn_count <= 0:
                state = FORWARD
        elif state == TRACKING and not traget_found:
            state = FORWARD


if __name__ == "__main__":
    main()
