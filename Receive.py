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

DEPTH_THRESHOLD = 30  # You can tweak this

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
    if depth_frame is None:
        return None
    depth_h, depth_w = depth_frame.shape[:2]
    x_c = int(lm.x * color_w)
    y_c = int(lm.y * color_h)
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

# ---------------- Wrist Contact Check -----------------
def check_wrist_contact(wx, wy, wz, hand_side, stomach_polygon, stomach_depth):
    """
    Returns True if:
        - Wrist is inside torso polygon (XY)
        - Wrist depth is near torso depth (Z)
        - Any finger is pressed
    """
    if not any(touch_state.values()):
        return False, None
    if stomach_polygon is None or stomach_depth is None:
        return False, None
    inside = cv2.pointPolygonTest(stomach_polygon, (wx, wy), False)
    if inside < 0:
        return False, None
    if abs(wz - stomach_depth) <= DEPTH_THRESHOLD:
        return True, f"{hand_side} wrist near torso & finger pressed"
    return False, None

# ---------------- Wrist Processing -----------------
def process_wrists(results_pose, stomach_polygon, stomach_depth, color_frame):
    contact_detected = False
    wrist_positions = []

    if results_pose.pose_landmarks:
        lm = results_pose.pose_landmarks.landmark
        for side in ["RIGHT_WRIST", "LEFT_WRIST"]:
            wrist_lm = lm[getattr(mp.solutions.pose.PoseLandmark, side)]
            xyz = get_pixel_depth(wrist_lm, color_frame.shape[1], color_frame.shape[0], latest_depth)
            if xyz is None:
                continue
            wx, wy, wz = xyz
            wrist_positions.append((wx, wy, wz))
            hand_side = "Left" if "LEFT" in side else "Right"

            contact, msg = check_wrist_contact(wx, wy, wz, hand_side, stomach_polygon, stomach_depth)
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

    return contact_detected, wrist_positions

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

            if color_frame is None or depth_frame is None:
                cv2.waitKey(1)
                continue

            frame_rgb = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
            depth_resized = cv2.resize(depth_frame, (color_frame.shape[1], color_frame.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)

            results_pose = pose.process(frame_rgb)
            compute_body_positions(results_pose, color_frame.shape[1], color_frame.shape[0], depth_resized)
            stomach_polygon, stomach_depth = compute_stomach_polygon(results_pose, color_frame.shape[1], color_frame.shape[0], depth_resized)

            if stomach_polygon is not None:
                cv2.polylines(color_frame, [stomach_polygon], True, (255, 255, 0), 2)

            process_wrists(results_pose, stomach_polygon, stomach_depth, color_frame)

            # Depth visualization
            depth_vis = cv2.normalize(depth_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

            combined = np.hstack((
                cv2.resize(color_frame, (640, 360)),
                cv2.resize(depth_vis, (640, 360))
            ))
            cv2.imshow("RGB + Depth + Wrist Detection", combined)

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
