# mediapipe_receive_wrist_contact.py
import socket
import cv2  # type: ignore
import numpy as np  # type: ignore
import struct
import mediapipe as mp  # type: ignore
import threading
import serial.tools.list_ports
import serial
import time
import pyvolume
import random


HOST = '127.0.0.1'
PORT = 5001

BAUD_RATE = 115200
latest_color = None
latest_depth = None
lock = threading.Lock()
slider_lock = threading.Lock()
slider_target = 50       # where the slider wants the volume to be
current_volume = 50      # what we have actually set
running = True
index_slider_value = current_volume

DEPTH_THRESHOLD = 200  # You can tweak this

last_contact_time = {
    "Left": 0,
    "Right": 0
}

# Touch state per finger (from Arduino)
touch_state = {
    "thumb_tip": False,
    "index_finger_tip": False,
    "middle_finger_tip": False,
    "ring_finger_tip": False,
    "pinky_tip": False
}

# Key joints for torso
BODY_JOINTS = [
    mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
    mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
    mp.solutions.pose.PoseLandmark.LEFT_HIP,
    mp.solutions.pose.PoseLandmark.RIGHT_HIP,
    mp.solutions.pose.PoseLandmark.NOSE
]

# ---------------- Simon Says -----------------
SIMON_SAYS_DURATION = 30  # seconds to respond
task_counter = 0
current_task = None
task_start_time = 0
task_active = False
task_completed = False

# All fingers and body parts for random tasks
FINGERS = ["thumb_tip", "index_finger_tip", "middle_finger_tip", "ring_finger_tip"]
BODY_PARTS = ["torso", "head", "shoulders", "left arm"]

# Modes
MODE_MENU = "menu"
MODE_SIMON = "simon_says"
MODE_VOLUME = "volume_control"

current_mode = MODE_MENU
mode_selected_time = 0
MODE_HOLD_THRESHOLD = 0.5  # seconds holding a body part to select mode


# ---------------- Serial Reading -----------------
def find_usb_serial_port():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        desc = port.description.lower()
        if "usb serial port" in desc:
            return port.device
    print("Arduino not found.")
    return None

def read_serial():
    global index_slider_value, running, touch_state, current_mode
    SERIAL_PORT = find_usb_serial_port()
    if SERIAL_PORT is None:
        return
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

    key_map = {
        "thumb": "thumb_tip",
        "index": "index_finger_tip",
        "middle": "middle_finger_tip",
        "ring": "ring_finger_tip",
        "pinky": "pinky_tip"
    }

    # smoothing buffer for sliders
    smoothed_slider = {k: 0.0 for k in key_map.values()}
    smoothed_slider2 = {k: current_volume for k in key_map.values()}
    ALPHA = 0.25  # lower = smoother (0.05–0.25 recommended)


    # Track slider state per finger
    slider_active = {k: False for k in key_map.values()}
    slider_value = {k: 0.0 for k in key_map.values()}

    global running, touch_state

    while running:
        try:
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                continue
            line_lower = line.lower()

            # Update touch state and slider activation
            for finger_name in key_map:
                key = key_map[finger_name]
                if finger_name in line_lower:
                    if "tapped" in line_lower:
                        slider_active[key] = True   # start slider
                    elif "lifted" in line_lower:
                        slider_active[key] = False  # end slider
                    touch_state[key] = "lifted" not in line_lower
                    break
                
                # Parse slider values
                if "slider" in line_lower:
                    parts = line.split(":")
                    if len(parts) == 2:
                        finger_part = parts[0].strip().lower()
                        value_part = parts[1].strip()
                        try:
                            val = float(value_part)
                            for finger_name in key_map:
                                key = key_map[finger_name]
                                if finger_name in finger_part and slider_active[key]:
                                    # LOW-PASS FILTER
                                    smoothed_slider[key] = smoothed_slider[key] * (1 - ALPHA) + val * ALPHA

                                    # Update slider2 for volume
                                    smoothed_slider2[key] = smoothed_slider[key]

                                    slider_value[key] = smoothed_slider[key]

                                    break
                                    # Update shared slider value for volume thread
                                with slider_lock:
                                    index_slider_value = smoothed_slider2["index_finger_tip"]
                                    index_slider_value = max(0, min(100, index_slider_value))
                        except ValueError:
                            continue

            
            # print(slider_value["index_finger_tip"]
            # Optional debug
            # print("Touch:", touch_state)
            # print("Sliders:", slider_value)
            print("Index slider:", slider_value["index_finger_tip"])
            # Update shared slider value instead of calling pyvolume directly
            # Inside your serial reading loop, after computing smoothed slider values:
            with slider_lock:
                if slider_active["index_finger_tip"]:
                    index_slider_value = smoothed_slider2["index_finger_tip"]
                    index_slider_value = max(0, min(100, index_slider_value))
                    # print(index_slider_value)

        except Exception as e:
            print("Serial error:", e)
            break
    ser.close()

def set_system_volume(vol):
    global current_mode

    if current_mode == MODE_VOLUME:
        vol = int(round(vol))   # round to nearest integer
        vol = max(0, min(100, vol))  # clamp just in case
        pyvolume.custom(vol)

def volume_thread():
    global current_volume, index_slider_value, running, current_mode

    step = 3       # volume step per update
    delay = 0.05   # seconds between steps

    while running:
        if current_mode == MODE_VOLUME:
            # safely get the target volume
            with slider_lock:
                target = max(0, min(100, index_slider_value))

            # ramp toward the target
            if current_volume < target:
                current_volume = min(current_volume + step, target)
                set_system_volume(current_volume)
            elif current_volume > target:
                current_volume = max(current_volume - step, target)
                set_system_volume(current_volume)
            # else: already at target, do nothing
        else:
            # Optional: keep the volume at the last value, or pause updates
            pass

        time.sleep(delay)


# ---------------- Frame Receiver -----------------
def recvall(sock, n):
    buf = b''
    while len(buf) < n:
        try:
            data = sock.recv(n - len(buf))
        except Exception as e:
            print("Socket recv error:", e)
            return None
        if not data:
            # Connection closed
            return None
        buf += data
    return buf


def receive_frames(conn):
    global latest_color, latest_depth, running
    while running:
        frame_type = recvall(conn, 1)
        if not frame_type:
            print("Frame type not received, stopping receive thread.")
            running = False
            break

        raw_len = recvall(conn, 4)
        if not raw_len:
            print("Frame length not received, stopping receive thread.")
            running = False
            break

        length = struct.unpack(">L", raw_len)[0]
        data = recvall(conn, length)
        if not data:
            print("Frame data incomplete, stopping receive thread.")
            running = False
            break

        if frame_type == b'C':
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                with lock:
                    latest_color = frame
        elif frame_type == b'D':
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
            if frame is not None:
                with lock:
                    latest_depth = frame


# ---------------- Utilities -----------------
def get_pixel_depth(lm, color_w, color_h, depth_frame):
    if depth_frame is None:
        return None
    depth_h, depth_w = depth_frame.shape[:2]
    x_c = int(lm.x * color_w)
    y_c = int(lm.y * color_h)
    x_d = int(x_c * depth_w / color_w)
    y_d = int(y_c * depth_h / color_h)
    if x_d < 0 or x_d >= depth_w or y_d < 0 or y_d >= depth_h:
        return None
    z = get_stable_depth(depth_frame, x_d, y_d, window=7)
    if z is None:
        return None
    return x_c, y_c, z

def get_stable_depth(depth_frame, x, y, window=7):
    h, w = depth_frame.shape
    half = window // 2

    xs = range(max(0, x-half), min(w, x+half+1))
    ys = range(max(0, y-half), min(h, y+half+1))

    samples = []
    for yy in ys:
        for xx in xs:
            d = depth_frame[yy, xx]
            if d > 0:              # ignore invalid
                samples.append(d)

    if len(samples) == 0:
        return None               # no valid depth nearby

    return np.median(samples)     # stable depth

def compute_body_positions(results_pose, color_w, color_h, depth_frame):
    positions = {}
    if results_pose.pose_landmarks:
        for joint in BODY_JOINTS:
            lm = results_pose.pose_landmarks.landmark[joint]
            pos = get_pixel_depth(lm, color_w, color_h, depth_frame)
            if pos:
                positions[joint] = pos
    return positions

def compute_stomach_polygon(results_pose, color_w, color_h, depth_frame):
    if not results_pose.pose_landmarks:
        return None, None
    lm = results_pose.pose_landmarks.landmark
    try:
        left_shoulder = get_pixel_depth(lm[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER], color_w, color_h, depth_frame)
        right_shoulder = get_pixel_depth(lm[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER], color_w, color_h, depth_frame)
        left_hip = get_pixel_depth(lm[mp.solutions.pose.PoseLandmark.LEFT_HIP], color_w, color_h, depth_frame)
        right_hip = get_pixel_depth(lm[mp.solutions.pose.PoseLandmark.RIGHT_HIP], color_w, color_h, depth_frame)
        if None in (left_shoulder, right_shoulder, left_hip, right_hip):
            return None, None
    except Exception:
        return None, None
    polygon = np.array([
        [left_shoulder[0], left_shoulder[1]],
        [right_shoulder[0], right_shoulder[1]],
        [right_hip[0], right_hip[1]],
        [left_hip[0], left_hip[1]]
    ])
    avg_depth = np.mean([left_shoulder[2], right_shoulder[2], left_hip[2], right_hip[2]])
    return polygon, avg_depth

def compute_head_polygon(results_pose, color_w, color_h, depth_frame, widen_factor=1.6, height_factor=8.0):
    if not results_pose.pose_landmarks:
        return None, None

    lm = results_pose.pose_landmarks.landmark

    # Base landmarks
    nose = get_pixel_depth(lm[mp.solutions.pose.PoseLandmark.NOSE], color_w, color_h, depth_frame)
    left_ear = get_pixel_depth(lm[mp.solutions.pose.PoseLandmark.LEFT_EAR], color_w, color_h, depth_frame)
    right_ear = get_pixel_depth(lm[mp.solutions.pose.PoseLandmark.RIGHT_EAR], color_w, color_h, depth_frame)

    # Chin from mouth midpoint
    ml = lm[mp.solutions.pose.PoseLandmark.MOUTH_LEFT]
    mr = lm[mp.solutions.pose.PoseLandmark.MOUTH_RIGHT]
    chin_lm = type(ml)()
    chin_lm.x = (ml.x + mr.x) / 2
    chin_lm.y = (ml.y + mr.y) / 2
    chin_lm.z = (ml.z + mr.z) / 2
    chin = get_pixel_depth(chin_lm, color_w, color_h, depth_frame)

    if None in (nose, left_ear, right_ear, chin):
        return None, None

    # Base head width from ears
    base_width = abs(right_ear[0] - left_ear[0])

    # Make the head polygon wider
    width = int(base_width * widen_factor)

    # Center X between ears
    cx = int((left_ear[0] + right_ear[0]) / 2)

    left_x  = cx - width // 2
    right_x = cx + width // 2

    # Height from nose-to-chin, scaled
    base_height = abs(chin[1] - nose[1])
    height = int(base_height * height_factor)

    # Top above nose by full height
    top_y = nose[1] - height

    # Bottom is chin
    bottom_y = chin[1]

    polygon = np.array([
        [cx,       top_y],     # Top
        [left_x,   nose[1]],   # Left
        [cx,       bottom_y],  # Bottom (chin)
        [right_x,  nose[1]]    # Right
    ])

    avg_depth = np.mean([nose[2], left_ear[2], right_ear[2], chin[2]])
    return polygon, avg_depth



def compute_shoulders_polygon(results_pose, color_w, color_h, depth_frame):
    if not results_pose.pose_landmarks:
        return None, None

    lm = results_pose.pose_landmarks.landmark
    left_shoulder = get_pixel_depth(lm[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER], color_w, color_h, depth_frame)
    right_shoulder = get_pixel_depth(lm[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER], color_w, color_h, depth_frame)

    if None in (left_shoulder, right_shoulder):
        return None, None

    # Make the polygon wider and "forward" along Y to enclose the shoulders fully
    horizontal_offset = 80  # width left/right
    vertical_offset = 40    # extend forward/back

    polygon = np.array([
        [left_shoulder[0] + horizontal_offset, left_shoulder[1] - vertical_offset],   # top-left
        [right_shoulder[0] - horizontal_offset, right_shoulder[1] - vertical_offset], # top-right
        [right_shoulder[0] - horizontal_offset, right_shoulder[1] + vertical_offset], # bottom-right
        [left_shoulder[0] + horizontal_offset, left_shoulder[1] + vertical_offset]    # bottom-left
    ])

    avg_depth = np.mean([left_shoulder[2], right_shoulder[2]])
    return polygon, avg_depth


def compute_arm_polygon(results_pose, color_w, color_h, depth_frame, side="Left"):
    if not results_pose.pose_landmarks:
        return None, None
    lm = results_pose.pose_landmarks.landmark
    if side=="Left":
        wrist_lm = lm[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
        elbow_lm = lm[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW]
        shoulder_lm = lm[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
    else:
        wrist_lm = lm[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
        elbow_lm = lm[mp.solutions.pose.PoseLandmark.LEFT_ELBOW]
        shoulder_lm = lm[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]

    wrist = get_pixel_depth(wrist_lm, color_w, color_h, depth_frame)
    elbow = get_pixel_depth(elbow_lm, color_w, color_h, depth_frame)
    shoulder = get_pixel_depth(shoulder_lm, color_w, color_h, depth_frame)
    if None in (wrist, elbow, shoulder):
        return None, None

    offset = 30
    polygon = np.array([
        [wrist[0]-offset, wrist[1]],
        [wrist[0]+offset, wrist[1]],
        [elbow[0]+offset, elbow[1]],
        [elbow[0]-offset, elbow[1]]
    ])
    avg_depth = np.mean([wrist[2], elbow[2], shoulder[2]])
    return polygon, avg_depth



# ---------------- Wrist Contact Check -----------------
CONTACT_HOLD_TIME = 0.150   # 150 ms of smoothing

def check_wrist_contact(wx, wy, wz, hand_side, polygons_with_depths):

    global last_contact_time, current_mode
    now = time.time()  # seconds (float)
    
    if current_mode == MODE_VOLUME:
        if touch_state.get("middle_finger_tip", False):
            for poly, poly_depth, name in polygons_with_depths:
                if poly is None or poly_depth is None:
                    continue
                if name.lower() == "torso":
                    inside = cv2.pointPolygonTest(poly, (wx, wy), False)
                    if inside >= 0 and abs(wz - poly_depth) <= DEPTH_THRESHOLD:
                        # Mute system volume
                        pyvolume.custom(0)
                        print("Middle finger tapped stomach → volume muted")
                        last_contact_time[hand_side] = now
                        return True, "middle_finger_tip", "torso"

    # If ANY finger is not pressed → no contact at all
    # if not any(touch_state.values()):
    #     # But still allow smoothing:
    #     if now - last_contact_time[hand_side] < CONTACT_HOLD_TIME:
    #         return True, f"{hand_side} wrist (held contact)"
    #     return False, None

    # # Evaluate new contact
    # for poly, poly_depth, name in polygons_with_depths:
    #     if poly is None or poly_depth is None:
    #         continue

    #     inside = cv2.pointPolygonTest(poly, (wx, wy), False)
    #     if inside >= 0 and abs(wz - poly_depth) <= DEPTH_THRESHOLD:
    #         # Register fresh contact
    #         last_contact_time[hand_side] = now
    #         return True, f"{hand_side} wrist near {name} & finger pressed"

    # # No new contact, but apply hysteresis
    # if now - last_contact_time[hand_side] < CONTACT_HOLD_TIME:
    #     return True, f"{hand_side} wrist (held contact)"

    # return False, None

    for finger, pressed in touch_state.items():
        if not pressed:
            continue

        # Check all polygons
        for poly, poly_depth, name in polygons_with_depths:
            if poly is None or poly_depth is None:
                continue

            inside = cv2.pointPolygonTest(poly, (wx, wy), False)
            if inside >= 0 and abs(wz - poly_depth) <= DEPTH_THRESHOLD:
                last_contact_time[hand_side] = now
                return True, finger, name  # finger pressed, touching this polygon

    # Apply hold-time smoothing if no new contact
    if now - last_contact_time[hand_side] < CONTACT_HOLD_TIME:
        return True, None, None

    return False, None, None

# ---------------- Physical Right Wrist Processing (with coords print) -----------------

def get_wrist_palm_xyz(lm,color_w, color_h, depth_frame, scale=-0.5):
    """
    Returns an adjusted wrist point closer to the palm.
    scale: fraction along wrist -> middle MCP joint
    """

    wrist_lm = lm[mp.solutions.pose.PoseLandmark.LEFT_WRIST]  # mirrored
    elbow_lm = lm[mp.solutions.pose.PoseLandmark.LEFT_ELBOW]
 
    wrist_xyz = get_pixel_depth(wrist_lm, color_w, color_h, depth_frame)
    elbow_xyz = get_pixel_depth(elbow_lm, color_w, color_h, depth_frame)

    if wrist_xyz is None or elbow_xyz is None:
        return wrist_xyz  # fallback to wrist if something fails

    # Vector from wrist to elbow
    vx = elbow_xyz[0] - wrist_xyz[0]
    vy = elbow_xyz[1] - wrist_xyz[1]
    vz = elbow_xyz[2] - wrist_xyz[2]

    # Offset point along that vector
    palm_xyz = (
        int(wrist_xyz[0] + vx * scale),
        int(wrist_xyz[1] + vy * scale),
        wrist_xyz[2] + vz * 0.3
    )

    return palm_xyz


def process_physical_right_wrist(results_pose, polygons_list, color_frame):
    contact_detected = False
    wrist_position = None
    finger = None  # initialize
    name = None    # initialize

    if results_pose.pose_landmarks:
        lm = results_pose.pose_landmarks.landmark
        xyz = get_wrist_palm_xyz(lm, color_frame.shape[1], color_frame.shape[0], latest_depth)
        if xyz is not None:
            wx, wy, wz = xyz
            wrist_position = (wx, wy, wz)
            hand_side = "Right"

            # Print wrist coordinates
            # print(f"[DEBUG] {hand_side} wrist coords: x={wx}, y={wy}, z={wz}")


            contact, finger, name = check_wrist_contact(
                wx, wy, wz, "Right", polygons_list
            )
            
            if contact:
                contact_detected = True
            #     # Large screen flash
                # overlay = color_frame.copy()
                # cv2.rectangle(overlay, (0, 0), (color_frame.shape[1], color_frame.shape[0]), (0, 0, 255), -1)
                # alpha = 0.7
                # cv2.addWeighted(overlay, alpha, color_frame, 1 - alpha, 0, color_frame)
            
            else:
                cv2.circle(color_frame, (wx, wy), 12, (0, 255, 0), 2)
                cv2.putText(color_frame, hand_side + " wrist", (wx + 10, wy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return contact_detected, wrist_position, finger, name

def start_new_task():
    global current_task, task_start_time, task_active, task_completed, task_counter
    if task_counter < 6: 
        finger = random.choice(FINGERS)
        body_part = random.choice(BODY_PARTS)
        current_task = (finger, body_part)
        task_start_time = time.time()
        task_active = True
        task_completed = False
        task_counter += 1
        print(task_counter)
        print(f"Simon Says: Touch your {body_part} with your {finger.replace('_tip','')}!")

def check_task_completion(contact_info):
    """
    Simple task completion check:
    Returns True if the current task matches the finger & body part in contact_info
    """
    global current_task
    if not current_task:
        return False

    finger, body_part = current_task
    contact_detected, contact_finger, contact_body = contact_info

    # No contact detected
    if not contact_detected:
        return False

    # Finger or body info is missing → cannot complete task
    if contact_finger is None or contact_body is None:
        return False

    # Compare with current task
    if contact_finger == finger and contact_body.lower() == body_part.lower():
        start_new_task()
        return True

    return False

# ---------------- Main -----------------
def main():
    global running, task_completed, task_active, current_mode, task_counter

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((HOST, PORT))
    sock.listen(1)
    print("Waiting for Kinect connection...")
    conn, addr = sock.accept()
    print(f"Connected to {addr}")

    threading.Thread(target=receive_frames, args=(conn,), daemon=True).start()
    threading.Thread(target=read_serial, daemon=True).start()
    threading.Thread(target=volume_thread, daemon=True).start()

    try:
        while running:
            with lock:
                color_frame = latest_color.copy() if latest_color is not None else None
                depth_frame = latest_depth.copy() if latest_depth is not None else None
            # Update shared slider value instead of calling pyvolume directly
        
            # Skip iteration if frames aren't ready yet
            if color_frame is None or depth_frame is None:
                cv2.waitKey(1)
                continue

            # Prepare frames
            frame_rgb = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
            depth_resized = cv2.resize(depth_frame, (color_frame.shape[1], color_frame.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)

            results_pose = pose.process(frame_rgb)

            # Torso / stomach
            stomach_polygon, stomach_depth = compute_stomach_polygon(
                results_pose, color_frame.shape[1], color_frame.shape[0], depth_resized
            )
            head_polygon, head_depth = compute_head_polygon(results_pose, color_frame.shape[1], color_frame.shape[0], depth_resized)
            shoulders_polygon, shoulders_depth = compute_shoulders_polygon(results_pose, color_frame.shape[1], color_frame.shape[0], depth_resized)
            right_arm_polygon, right_arm_depth = compute_arm_polygon(results_pose, color_frame.shape[1], color_frame.shape[0], depth_resized, side="Right")
            left_arm_polygon, left_arm_depth = compute_arm_polygon(results_pose, color_frame.shape[1], color_frame.shape[0], depth_resized, side="Left")

            polygons_list = [
                (stomach_polygon, stomach_depth, "torso"),
                (head_polygon, head_depth, "head"),
                (shoulders_polygon, shoulders_depth, "shoulders"),
                # (right_arm_polygon, right_arm_depth, "right arm"),
                (left_arm_polygon, left_arm_depth, "left arm")
            ]
            
            # print("[DEBUG] Head Depth:", head_depth)
            # print("[DEBUG] shoulders_depth:", shoulders_depth)
            # print("[DEBUG] right_arm_depth:", right_arm_depth)
            # print("[DEBUG] left_arm_depth:", left_arm_depth)
            # print("[DEBUG] stomach depth:", stomach_depth)

            if stomach_polygon is not None:
                cv2.polylines(color_frame, [stomach_polygon], True, (255, 255, 0), 2)
            if head_polygon is not None:
                cv2.polylines(color_frame, [head_polygon], True, (0, 255, 255), 2)
            if shoulders_polygon is not None:
                cv2.polylines(color_frame, [shoulders_polygon], False, (255, 255, 0), 2)
            # if right_arm_polygon is not None:
            #     cv2.polylines(color_frame, [right_arm_polygon], True, (0, 0, 255), 2)
            if left_arm_polygon is not None:
                cv2.polylines(color_frame, [left_arm_polygon], True, (0, 255, 0), 2)


            contact, _, finger, body_part = process_physical_right_wrist(results_pose, polygons_list, color_frame)

            #print(current_mode)
            if current_mode == MODE_MENU:
                overlay = color_frame.copy()
                alpha = 0.6
                cv2.rectangle(overlay, (50, 50), (color_frame.shape[1]-50, 250), (50, 50, 50), -1)
                cv2.addWeighted(overlay, alpha, color_frame, 1 - alpha, 0, color_frame)

                # Menu title
                cv2.putText(color_frame, "MAIN MENU", (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)

                # Instructions
                cv2.putText(color_frame, "Touch your torso -> Simon Says", (60, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
                cv2.putText(color_frame, "Touch your shoulders -> Volume Control", (60, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)               
                
                if contact and body_part is not None:
                    if body_part.lower() == "torso":
                        current_mode = MODE_SIMON
                        task_active = False
                        task_counter = 0
                        print("Entering Simon Says mode!")
                    elif body_part.lower() == "shoulders":
                        current_mode = MODE_VOLUME
                        print("Entering Volume Control mode!")

            # ---------------- Simon Says Mode ----------------
            elif current_mode == MODE_SIMON:
                if not task_active or time.time() - task_start_time > SIMON_SAYS_DURATION:
                    start_new_task()

                contact_info = (contact, finger, body_part)
                task_completed = check_task_completion(contact_info)

                if task_active and not task_completed:
                    finger_name, body_target = current_task
                    cv2.putText(color_frame, f"Simon Says: {finger_name.replace('_tip','')} -> {body_target}",
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 2)
                    elapsed = time.time() - task_start_time
                    remaining = max(0, int(SIMON_SAYS_DURATION - elapsed))
                    cv2.putText(color_frame, f"Time left: {remaining}s",
                                (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                if task_completed:
                    overlay = color_frame.copy()
                    cv2.rectangle(overlay, (0,0), (color_frame.shape[1], color_frame.shape[0]), (0,255,0), -1)
                    cv2.addWeighted(overlay, 0.9, color_frame, 0.1, 0, color_frame)
                    cv2.putText(color_frame, "TASK COMPLETE!", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)

                if task_counter >= 6:
                    # Game over
                    overlay = color_frame.copy()
                    cv2.rectangle(overlay, (0,0), (color_frame.shape[1], color_frame.shape[0]), (0,0,255), -1)
                    cv2.addWeighted(overlay, 0.9, color_frame, 0.1, 0, color_frame)
                    cv2.putText(color_frame, "GAME OVER!", (color_frame.shape[1]//4, color_frame.shape[0]//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 5)
                    current_mode = MODE_MENU

            # ---------------- Volume Control Mode ----------------
            elif current_mode == MODE_VOLUME:
                # volume_thread is already updating system volume
                cv2.putText(color_frame, f"Volume Control Mode", (50,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,0), 3)
                cv2.putText(color_frame, f"Current Volume: {current_volume}", (50,100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,0), 2)

            # # Depth visualization
            # depth_vis = cv2.normalize(depth_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            # depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

            # Combine color and depth
            combined = np.hstack((
                cv2.resize(color_frame, (1280, 720)),
                # cv2.resize(depth_vis, (640, 360))
            ))
            cv2.imshow("RGB + Body Detection", combined)

            if cv2.waitKey(1) & 0xFF == 27:
                running = False
                break



    finally:
        running = False
        conn.close()
        sock.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
