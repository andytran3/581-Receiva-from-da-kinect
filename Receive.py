# mediapipe_receive_wrist_contact.py
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

DEPTH_THRESHOLD = 200  # You can tweak this

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
            for finger_name in key_map:
                if finger_name in line_lower:
                    fingertip_key = key_map[finger_name]
                    touch_state[fingertip_key] = True
                    if "lifted" in line_lower:
                        touch_state[fingertip_key] = False
                    break
            print(touch_state)
        except Exception as e:
            print("Serial error:", e)
            break
    ser.close()

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

def compute_head_position(results_pose, color_w, color_h, depth_frame):
    if not results_pose.pose_landmarks:
        return None, None
    lm = results_pose.pose_landmarks.landmark
    try:
        nose = get_pixel_depth(lm[mp.solutions.pose.PoseLandmark.NOSE], color_w, color_h, depth_frame)
        left_ear = get_pixel_depth(lm[mp.solutions.pose.PoseLandmark.LEFT_EAR], color_w, color_h, depth_frame)
        right_ear = get_pixel_depth(lm[mp.solutions.pose.PoseLandmark.RIGHT_EAR], color_w, color_h, depth_frame)
        if None in (nose, left_ear, right_ear):
            return None, None
    except Exception:
        return None, None

    polygon = np.array([
        [left_ear[0], left_ear[1]],
        [right_ear[0], right_ear[1]],
        [nose[0], nose[1]]
    ])
    avg_depth = np.mean([nose[2], left_ear[2], right_ear[2]])
    return polygon, avg_depth

def compute_shoulders_polygon(results_pose, color_w, color_h, depth_frame):
    if not results_pose.pose_landmarks:
        return None, None
    lm = results_pose.pose_landmarks.landmark
    try:
        left_shoulder = get_pixel_depth(lm[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER], color_w, color_h, depth_frame)
        right_shoulder = get_pixel_depth(lm[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER], color_w, color_h, depth_frame)
        if None in (left_shoulder, right_shoulder):
            return None, None
    except Exception:
        return None, None

    polygon = np.array([
        [left_shoulder[0], left_shoulder[1]],
        [right_shoulder[0], right_shoulder[1]]
    ])
    avg_depth = np.mean([left_shoulder[2], right_shoulder[2]])
    return polygon, avg_depth

def compute_arm_polygon(results_pose, color_w, color_h, depth_frame, side="Left"):
    """
    Compute a polygon from wrist → elbow → shoulder for one arm.
    side: "Left" or "Right" (physical sides, mirrored if needed)
    Returns: polygon (4 points) and average depth
    """
    if not results_pose.pose_landmarks:
        return None, None

    lm = results_pose.pose_landmarks.landmark

    # Map physical side to MediaPipe joints (mirrored feed)
    if side == "Left":
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

    # Compute a simple "arm polygon" using the 3 points
    # We'll make it a quadrilateral by offsetting slightly along X or Y if needed
    polygon = np.array([
        [wrist[0], wrist[1]],
        [elbow[0], elbow[1]],
        [shoulder[0], shoulder[1]],
        [elbow[0], elbow[1]]  # duplicate elbow to make quad for simplicity
    ])

    avg_depth = np.mean([wrist[2], elbow[2], shoulder[2]])
    return polygon, avg_depth


# ---------------- Wrist Contact Check -----------------
def check_wrist_contact(wx, wy, wz, hand_side, polygons_with_depths):
    """
    Returns True if:
        - Wrist is inside any polygon (XY)
        - Wrist depth is near that polygon's depth (Z)
        - Any finger is pressed
    polygons_with_depths: list of tuples [(polygon, avg_depth, name_str), ...]
    """
    if not any(touch_state.values()):
        return False, None

    for poly, poly_depth, name in polygons_with_depths:
        if poly is None or poly_depth is None:
            continue
        inside = cv2.pointPolygonTest(poly, (wx, wy), False)
        if inside >= 0 and abs(wz - poly_depth) <= DEPTH_THRESHOLD:
            return True, f"{hand_side} wrist near {name} & finger pressed"

    return False, None


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
        wrist_xyz[2] + vz * scale  # optional: adjust depth slightly
    )

    return palm_xyz


def process_physical_right_wrist(results_pose, polygons_list, color_frame):
    contact_detected = False
    wrist_position = None

    if results_pose.pose_landmarks:
        lm = results_pose.pose_landmarks.landmark
        xyz = get_wrist_palm_xyz(lm, color_frame.shape[1], color_frame.shape[0], latest_depth)
        if xyz is not None:
            wx, wy, wz = xyz
            wrist_position = (wx, wy, wz)
            hand_side = "Right"

            # Print wrist coordinates
            print(f"[DEBUG] {hand_side} wrist coords: x={wx}, y={wy}, z={wz}")


            contact, msg = check_wrist_contact(
                wx, wy, wz, "Right", polygons_list
            )

            if contact:
                contact_detected = True
                # Large screen flash
                overlay = color_frame.copy()
                cv2.rectangle(overlay, (0, 0), (color_frame.shape[1], color_frame.shape[0]), (0, 0, 255), -1)
                alpha = 0.7
                cv2.addWeighted(overlay, alpha, color_frame, 1 - alpha, 0, color_frame)
                cv2.putText(color_frame, msg, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            else:
                cv2.circle(color_frame, (wx, wy), 12, (0, 255, 0), 2)
                cv2.putText(color_frame, hand_side + " wrist", (wx + 10, wy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return contact_detected, wrist_position




# ---------------- Main -----------------
def main():
    global running

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

    try:
        while running:
            with lock:
                color_frame = latest_color.copy() if latest_color is not None else None
                depth_frame = latest_depth.copy() if latest_depth is not None else None

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
            head_polygon, head_depth = compute_head_position(results_pose, color_frame.shape[1], color_frame.shape[0], depth_resized)
            shoulders_polygon, shoulders_depth = compute_shoulders_polygon(results_pose, color_frame.shape[1], color_frame.shape[0], depth_resized)
            right_arm_polygon, right_arm_depth = compute_arm_polygon(results_pose, color_frame.shape[1], color_frame.shape[0], depth_resized, side="Right")
            left_arm_polygon, left_arm_depth = compute_arm_polygon(results_pose, color_frame.shape[1], color_frame.shape[0], depth_resized, side="Left")

            polygons_list = [
                (stomach_polygon, stomach_depth, "torso"),
                (head_polygon, head_depth, "head"),
                (shoulders_polygon, shoulders_depth, "shoulders"),
                (right_arm_polygon, right_arm_depth, "right arm"),
                (left_arm_polygon, left_arm_depth, "left arm")
            ]
            
            print("[DEBUG] Head Depth:", head_depth)
            print("[DEBUG] shoulders_depth:", shoulders_depth)
            print("[DEBUG] right_arm_depth:", right_arm_depth)
            print("[DEBUG] left_arm_depth:", left_arm_depth)
            print("[DEBUG] stomach depth:", stomach_depth)

            if stomach_polygon is not None:
                cv2.polylines(color_frame, [stomach_polygon], True, (255, 255, 0), 2)
            if head_polygon is not None:
                cv2.polylines(color_frame, [head_polygon], True, (0, 255, 255), 2)
            if shoulders_polygon is not None:
                cv2.polylines(color_frame, [shoulders_polygon], False, (255, 255, 0), 2)
            if right_arm_polygon is not None:
                cv2.polylines(color_frame, [right_arm_polygon], True, (0, 0, 255), 2)
            if left_arm_polygon is not None:
                cv2.polylines(color_frame, [left_arm_polygon], True, (0, 255, 0), 2)


            process_physical_right_wrist(results_pose, polygons_list, color_frame)

            # Depth visualization
            depth_vis = cv2.normalize(depth_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

            # Combine color and depth
            combined = np.hstack((
                cv2.resize(color_frame, (640, 360)),
                cv2.resize(depth_vis, (640, 360))
            ))
            cv2.imshow("RGB + Depth + Body Detection", combined)

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
