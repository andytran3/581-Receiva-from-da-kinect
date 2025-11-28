# mediapipe_receive_touch_fixed.py
import socket
import cv2  # type: ignore
import numpy as np  # type: ignore
import struct
import mediapipe as mp  # type: ignore
import threading
import serial.tools.list_ports
import serial

HOST = '127.0.0.1'
PORT = 5001

BAUD_RATE = 115200
latest_color = None
latest_depth = None
lock = threading.Lock()
running = True

# Touch state per finger
touch_state = {
    "thumb_tip": False,
    "index_finger_tip": False,
    "middle_finger_tip": False,
    "ring_finger_tip": False,
    "pinky_tip": False
}

# Key joints
BODY_JOINTS = [
    mp.solutions.pose.PoseLandmark.LEFT_WRIST,
    mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
    mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
    mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
    mp.solutions.pose.PoseLandmark.NOSE
]

FINGER_TIPS = [
    mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
    mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP,
    mp.solutions.hands.HandLandmark.RING_FINGER_TIP,
    mp.solutions.hands.HandLandmark.PINKY_TIP
]

# ---------------- Serial Reading -----------------
def find_usb_serial_port():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        desc = port.description.lower()
        if "usb serial port" in desc:
            return port.device
    print("Arduino not found.")
    return None

def _reset_touch(fingertip_key):
    touch_state[fingertip_key] = False

def read_serial():
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
    global running
    while running:
        try:
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                continue
            line_lower = line.lower()
            print(line_lower)
            for finger_name in key_map:
                if finger_name in line_lower:
                    fingertip_key = key_map[finger_name]
                    touch_state[fingertip_key] = True
                    if "lifted" in line_lower: 
                        touch_state[fingertip_key] = False
                    # threading.Timer(0.05, lambda k=fingertip_key: _reset_touch(k)).start()
                    # print(touch_state)
                    break
        except Exception as e:
            print("Serial error:", e)
            break
    ser.close()

# ---------------- Frame Receiver -----------------
def recvall(sock, n):
    buf = b''
    while len(buf) < n:
        data = sock.recv(n - len(buf))
        if not data:
            return None
        buf += data
    return buf

def receive_frames(conn):
    global latest_color, latest_depth, running
    while running:
        frame_type = recvall(conn, 1)
        if not frame_type:
            running = False
            break
        raw_len = recvall(conn, 4)
        if not raw_len:
            running = False
            break
        length = struct.unpack(">L", raw_len)[0]
        data = recvall(conn, length)
        if not data:
            running = False
            break
        if frame_type == b'C':
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            with lock:
                latest_color = frame
        elif frame_type == b'D':
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
            with lock:
                latest_depth = frame

# ---------------- Utilities -----------------
def get_pixel_depth(lm, color_w, color_h, depth_frame):
    """Map normalized landmark to color-frame coordinates + depth safely."""
    if depth_frame is None:
        return None
    depth_h, depth_w = depth_frame.shape[:2]
    x_c = int(lm.x * color_w)
    y_c = int(lm.y * color_h)
    # Map color-frame pixels to depth frame
    x_d = int(x_c * depth_w / color_w)
    y_d = int(y_c * depth_h / color_h)
    if x_d < 0 or x_d >= depth_w or y_d < 0 or y_d >= depth_h:
        return None
    z = float(depth_frame[y_d, x_d])
    return x_c, y_c, z

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

def get_hand_sides(results_hands):
    sides = []
    if results_hands.multi_handedness:
        for handedness in results_hands.multi_handedness:
            sides.append(handedness.classification[0].label)
    return sides

def check_contact(hx, hy, hz, hand_side, fingertip_name, body_positions, stomach_polygon, stomach_depth, debug=False):
    """
    Returns True if:
      - The fingertip is near a body joint or stomach polygon (MediaPipe + depth), AND
      - Arduino reports this fingertip is touching (normal mode) OR
        ANY Arduino fingertip is pressed (debug mode)
    """

    # Determine if touch is active based on mode
    if debug:
        # In debug mode, allow any pressed sensor to enable contact
        if not any(touch_state.values()):
            return False, None
    else:
        # Normal mode: only consider this specific fingertip
        if not touch_state.get(fingertip_name, False):
            return False, None

    # Check body joints
    for bj, (bx, by, bz) in body_positions.items():
        joint_name = mp.solutions.pose.PoseLandmark(bj).name.lower()

        # Skip same-side joints
        if (hand_side == "Left" and "left" in joint_name) or \
           (hand_side == "Right" and "right" in joint_name):
            continue

        dx, dy, dz = hx - bx, hy - by, hz - bz
        if np.sqrt(dx**2 + dy**2) < 25 and abs(dz - bz) < 25:
            return True, f"{hand_side} {fingertip_name} touching {joint_name}"

    # Check stomach polygon
    if stomach_polygon is not None:
        inside = cv2.pointPolygonTest(stomach_polygon, (hx, hy), False)
        if inside >= 0 and abs(hz - stomach_depth) < 25:
            return True, f"{hand_side} {fingertip_name} touching stomach"

    return False, None


def detect_marker_centroids(frame, color_lower, color_upper):
    """
    Detects colored fingertip markers and returns their centroids.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, color_lower, color_upper)
    
    # clean mask
    mask = cv2.medianBlur(mask, 7)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroids = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 50:  # filter noise
            continue
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])
            centroids.append((cx, cy))
    return 

def match_fingers_to_markers(hand_landmarks, frame_width, frame_height, markers):
    """
    Assigns each marker to the closest fingertip.
    Returns dict: {"INDEX": (mx, my) or None, ...}
    """
    # fingertip IDs: Thumb 4, Index 8, Middle 12, Ring 16, Pinky 20
    tip_ids = {
        "THUMB": 4,
        "INDEX": 8,
        "MIDDLE": 12,
        "RING": 16,
        "PINKY": 20
    }

    results = {name: None for name in tip_ids}

    # Convert MediaPipe normalized coords → pixel coords
    fingertips = {}
    for name, idx in tip_ids.items():
        lm = hand_landmarks.landmark[idx]
        fx = int(lm.x * frame_width)
        fy = int(lm.y * frame_height)
        fingertips[name] = (fx, fy)

    # For each marker → assign to nearest fingertip
    for mx, my in markers:
        best_finger = None
        best_dist = 99999
        for name, (fx, fy) in fingertips.items():
            d = (fx - mx)**2 + (fy - my)**2
            if d < best_dist:
                best_dist = d
                best_finger = name
        results[best_finger] = (mx, my)

    return results

# ---------------- Glove Preprocessing -----------------
def preprocess_for_glove(frame):
    """
    Enhances visibility of black gloves before sending to MediaPipe.
    Includes contrast boost, sharpening, and optional color pop.
    """

    # 1. Convert to LAB → boost contrast on L-channel (lightness)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    lab = cv2.merge((l2, a, b))
    boosted = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 2. Sharpen edges
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharp = cv2.filter2D(boosted, -1, kernel)

    # OPTIONAL 3. Increase saturation (helps if colored stickers are used)
    hsv = cv2.cvtColor(sharp, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.add(s, 30)                # slight color pop
    v = cv2.add(v, 20)                # brighten highlights
    hsv = cv2.merge((h, s, v))
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return enhanced

def process_hands(results_hands, hand_sides, body_positions, stomach_polygon, stomach_depth, color_frame):
    """
    Process detected hands and fingertips, combining MediaPipe detection with Arduino fingertip press info.
    Returns:
        contact_detected (bool): True if any fingertip is touching a body part
        hand_joints (list): All fingertip coordinates for visualization
    """
    contact_detected = False
    hand_joints = []

    if results_hands.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
            hand_side = hand_sides[i] if i < len(hand_sides) else "Unknown"

            # Draw hand skeleton
            mp.solutions.drawing_utils.draw_landmarks(
                color_frame,
                hand_landmarks,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2),
                mp.solutions.drawing_utils.DrawingSpec(color=(0,255,255), thickness=2)
            )

            for tip in FINGER_TIPS:
                lm = hand_landmarks.landmark[tip]

                # Map fingertip enum → string key
                fingertip_name = mp.solutions.hands.HandLandmark(tip).name.lower()

                # Skip if Arduino says this finger is not pressed
                if not touch_state.get(fingertip_name, False):
                    continue
                    
                print("FINGER TIP IS TOUCHES YAYAYA:", fingertip_name)

                # SAFE depth lookup
                xyz = get_pixel_depth(lm, color_frame.shape[1], color_frame.shape[0], latest_depth)
                if xyz is None:
                    continue
                hx, hy, hz = xyz

                hand_joints.append((hx, hy, hz))

                # Check if fingertip is near a body joint or stomach polygon
                contact, msg = check_contact(hx, hy, hz, hand_side, fingertip_name, body_positions, stomach_polygon, stomach_depth, debug=True)
                if contact:
                    contact_detected = True
                    # Highlight fingertip
                    color = (0, 255, 255) if "stomach" in msg else (0, 0, 255)
                    cv2.circle(color_frame, (hx, hy), 12, color, -1)
                    cv2.putText(color_frame, msg, (50, 50 if "stomach" not in msg else 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    # Stop checking more fingertips if one is touching
                    break

            if contact_detected:
                break

    return contact_detected, hand_joints

# ---------------- Main -----------------
def main():
    global running

    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((HOST, PORT))
    sock.listen(1)
    print("Waiting for Kinect connection...")
    conn, addr = sock.accept()
    print(f"Connected to {addr}")

    threading.Thread(target=receive_frames, args=(conn,), daemon=True).start()
    threading.Thread(target=read_serial, daemon=True).start()

    try:
        while running:
            with lock:
                color_frame = latest_color.copy() if latest_color is not None else None
                depth_frame = latest_depth.copy() if latest_depth is not None else None

            if color_frame is None or depth_frame is None:
                cv2.waitKey(1)
                continue

            # -------------------------
            # 1. PREPROCESS GLOVE FRAME
            # -------------------------
            preprocessed = preprocess_for_glove(color_frame)
            frame_rgb = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)

            depth_resized = cv2.resize(
                depth_frame,
                (color_frame.shape[1], color_frame.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

            # -------------------------
            # 2. RUN MEDIAPIPE
            # -------------------------
            results_pose = pose.process(frame_rgb)
            results_hands = hands.process(frame_rgb)

            body_positions = compute_body_positions(
                results_pose, color_frame.shape[1], color_frame.shape[0], depth_resized
            )
            stomach_polygon, stomach_depth = compute_stomach_polygon(
                results_pose, color_frame.shape[1], color_frame.shape[0], depth_resized
            )

            if stomach_polygon is not None:
                cv2.polylines(color_frame, [stomach_polygon], True, (255, 255, 0), 2)

            hand_sides = get_hand_sides(results_hands)
            contact_detected, hand_joints = process_hands(
                results_hands, hand_sides, body_positions, stomach_polygon, stomach_depth, color_frame
            )

            # -------------------------
            # 3. DETECT YELLOW MARKERS
            # -------------------------
            markers = detect_marker_centroids(
                color_frame,
                color_lower=(25, 80, 100),
                color_upper=(35, 255, 255)
            ) or []

            # If markers exist, draw them
            for (mx, my) in markers:
                cv2.circle(color_frame, (mx, my), 6, (0, 255, 255), -1)

            # -------------------------
            # 4. MATCH MARKERS → FINGERTIPS
            # -------------------------
            if results_hands and results_hands.multi_hand_landmarks and markers:
                for hand_landmarks in results_hands.multi_hand_landmarks:

                    mapping = match_fingers_to_markers(
                        hand_landmarks,
                        frame_width=color_frame.shape[1],
                        frame_height=color_frame.shape[0],
                        markers=markers
                    )

                    # Visualize fingertip marker matches
                    for finger, pos in mapping.items():
                        if pos is not None:
                            mx, my = pos
                            cv2.putText(
                                color_frame,
                                f"{finger}",
                                (mx + 10, my),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 255),
                                1
                            )
                            cv2.circle(color_frame, (mx, my), 10, (0, 255, 255), 2)

                            # Log detected touches
                            print(f"Marker matched to {finger} at {pos}")

            # -------------------------
            # 5. ARDUINO TOUCH VISUALIZATION
            # -------------------------
            for i, (hx, hy, hz) in enumerate(hand_joints):
                fingertip_name = (
                    mp.solutions.hands.HandLandmark(FINGER_TIPS[i]).name.lower()
                    if i < len(FINGER_TIPS) else None
                )

                if fingertip_name and touch_state.get(fingertip_name, False):
                    color = (0, 255, 255) if contact_detected else (0, 255, 0)
                    radius = 10 if contact_detected else 5
                    cv2.circle(color_frame, (hx, hy), radius, color, -1)

                if contact_detected:
                    overlay = color_frame.copy()
                    cv2.rectangle(
                        overlay,
                        (0, 0),
                        (color_frame.shape[1], color_frame.shape[0]),
                        (0, 0, 255),
                        -1
                    )
                    alpha = 0.3
                    cv2.addWeighted(overlay, alpha, color_frame, 1 - alpha, 0, color_frame)

                    cv2.putText(
                        color_frame,
                        "HAND TOUCHING BODY!",
                        (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2
                    )

            # -------------------------
            # 6. DEPTH VISUALIZATION
            # -------------------------
            depth_vis = cv2.normalize(depth_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

            combined = np.hstack((
                cv2.resize(color_frame, (640, 360)),
                cv2.resize(depth_vis, (640, 360))
            ))
            cv2.imshow("RGB + Depth + Touch Detection", combined)

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
