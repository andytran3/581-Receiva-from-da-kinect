# mediapipe_receive_touch_fixed.py
import socket
import cv2 # type: ignore
import numpy as np # type: ignore
import struct
import mediapipe as mp # type: ignore
import threading
import serial.tools.list_ports
import serial

HOST = '127.0.0.1'
PORT = 5001

# reading from IMU
BAUD_RATE = 115200

# Shared frames and lock
latest_color = None
latest_depth = None
lock = threading.Lock()
running = True

# --- Hand-to-body contact detection with side-awareness ---
contact_detected = False

# Threshold map for different contact regions
THRESHOLD_MAP = {
    "hand_to_hand": 10,
    "hand_to_face": 25,
    "hand_to_torso": 40,
    "hand_to_leg": 35,
}
TOUCH_THRESHOLD = 25

# --- Key joints for detection ---
BODY_JOINTS = [
    mp.solutions.pose.PoseLandmark.LEFT_WRIST,
    mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
    mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
    mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
    mp.solutions.pose.PoseLandmark.NOSE
]

HAND_JOINTS = [
    mp.solutions.hands.HandLandmark.WRIST,
    mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
    mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP
]

FINGER_TIPS = [
    mp.solutions.HandLandmark.THUMB_TIP,
    mp.solutions.HandLandmark.INDEX_FINGER_TIP,
    mp.solutions.HandLandmark.MIDDLE_FINGER_TIP,
    mp.solutions.HandLandmark.RING_FINGER_TIP,
    mp.solutions.HandLandmark.PINKY_TIP
]


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
        print("No Arduino found.")
        return

    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

    while running:
        try:
            cur = ser.readline().decode(errors="ignore").strip()
            if cur:
                print("[SERIAL]", cur)
        except Exception as e:
            break

    ser.close()

# ----------------- Utilities -----------------
def recvall(sock, n):
    buf = b''
    while len(buf) < n:
        data = sock.recv(n - len(buf))
        if not data:
            return None
        buf += data
    return buf

def get_pixel_depth(lm, w, h, depth_frame):
    """Convert normalized landmark to pixel coordinates and get depth safely."""
    x = min(max(int(lm.x * w), 0), w - 1)
    y = min(max(int(lm.y * h), 0), h - 1)
    z = depth_frame[y, x]
    return x, y, z

# ----------------- Frame Receiver Thread -----------------
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

# ----------------- Main Program -----------------
def main():
    global running

    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=2,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((HOST, PORT))
    sock.listen(1)
    print("Waiting for Kinect connection...")
    conn, addr = sock.accept()
    print(f"Connected to {addr}")

    recv_thread = threading.Thread(target=receive_frames, args=(conn,), daemon=True)
    serial_thread = threading.Thread(target=read_serial, daemon=True)
    recv_thread.start()
    serial_thread.start()

    try:
        while running:
            with lock:
                color_frame = latest_color.copy() if latest_color is not None else None
                depth_frame = latest_depth.copy() if latest_depth is not None else None

            if color_frame is None or depth_frame is None:
                cv2.waitKey(1)
                continue

            frame_rgb = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
            h, w, _ = color_frame.shape
            depth_resized = cv2.resize(depth_frame, (w, h), interpolation=cv2.INTER_NEAREST)

            # --- Pose Detection ---
            results_pose = pose.process(frame_rgb)
            body_joints = []

            if results_pose.pose_landmarks:
                mp_drawing.draw_landmarks(
                    color_frame,
                    results_pose.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
                for joint in BODY_JOINTS:
                    lm = results_pose.pose_landmarks.landmark[joint]
                    bx, by, bz = get_pixel_depth(lm, w, h, depth_resized)
                    body_joints.append((bx, by, bz))

            # --- Hand Detection ---
            hand_joints = []
            results_hands = hands.process(frame_rgb)
            if results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        color_frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                    )
                    for joint in HAND_JOINTS:
                        lm = hand_landmarks.landmark[joint]
                        hx, hy, hz = get_pixel_depth(lm, w, h, depth_resized)
                        hand_joints.append((hx, hy, hz))

            # --- Hand-to-body contact detection ---
            # Get pose landmarks for later reference
            contact_detected = False
            pose_landmarks = results_pose.pose_landmarks.landmark if results_pose.pose_landmarks else None

            # Determine handedness info from Mediapipe
            hand_sides = []  # e.g., ["Left", "Right"]
            if results_hands.multi_handedness:
                for handedness in results_hands.multi_handedness:
                    hand_sides.append(handedness.classification[0].label)  # "Left" or "Right"

            # Iterate through each detected hand
            if results_hands.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
                    hand_side = hand_sides[i] if i < len(hand_sides) else "Unknown"

                    # Check each fingertip instead of every joint
                    for tip in FINGER_TIPS:
                        lm = hand_landmarks.landmark[tip]
                        hx, hy, hz = get_pixel_depth(lm, w, h, depth_resized)

                        fingertip_name = mp_hands.HandLandmark(tip).name.lower()

                        # Compare fingertip with all body joints
                        for bj, (bx, by, bz) in zip(BODY_JOINTS, body_joints):
                            joint_name = mp_pose.PoseLandmark(bj).name.lower()

                            # Skip same-side self-contact
                            if hand_side == "Left" and "left" in joint_name:
                                continue
                            if hand_side == "Right" and "right" in joint_name:
                                continue

                            # Dynamic threshold selection
                            if "shoulder" in joint_name or "chest" in joint_name:
                                threshold = THRESHOLD_MAP["hand_to_torso"]
                            elif "hip" in joint_name or "waist" in joint_name:
                                threshold = THRESHOLD_MAP["hand_to_leg"]
                            elif "nose" in joint_name or "face" in joint_name or "ear" in joint_name:
                                threshold = THRESHOLD_MAP["hand_to_face"]
                            elif "wrist" in joint_name or "hand" in joint_name:
                                threshold = THRESHOLD_MAP["hand_to_hand"]
                            else:
                                threshold = TOUCH_THRESHOLD

                            # Distance calculation
                            dx = hx - bx
                            dy = hy - by
                            dz = int(hz) - int(bz)
                            dist = np.sqrt(dx**2 + dy**2 + dz**2)

                            # Contact detected for THIS fingertip
                            if dist < threshold:
                                contact_detected = True

                                cv2.circle(color_frame, (hx, hy), 10, (0, 0, 255), -1)
                                cv2.putText(color_frame,
                                            f"{hand_side} {fingertip_name} touching {joint_name}",
                                            (50, 50),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                                break  # stop checking other body joints

                        if contact_detected:
                            break  # stop checking other fingertips

                    if contact_detected:
                        break  # stop checking other hands

            # --- Stomach region detection ---
            if results_pose.pose_landmarks:
                lm = results_pose.pose_landmarks.landmark
                ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
                rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                lh = lm[mp_pose.PoseLandmark.LEFT_HIP]
                rh = lm[mp_pose.PoseLandmark.RIGHT_HIP]

                # Convert normalized coords to pixel positions
                def to_px(lm):
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    z = depth_resized[y, x] if 0 <= y < h and 0 <= x < w else 0
                    return (x, y, z)

                left_shoulder = to_px(ls)
                right_shoulder = to_px(rs)
                left_hip = to_px(lh)
                right_hip = to_px(rh)

                # Define quadrilateral for stomach area
                stomach_polygon = np.array([
                    [left_shoulder[0], left_shoulder[1]],
                    [right_shoulder[0], right_shoulder[1]],
                    [right_hip[0], right_hip[1]],
                    [left_hip[0], left_hip[1]]
                ])

                # Optional visualization
                cv2.polylines(color_frame, [stomach_polygon], True, (255, 255, 0), 2)

                # --- Hand to stomach contact detection ---
                for hx, hy, hz in hand_joints:
                    # Check if (hx, hy) is inside stomach polygon
                    inside = cv2.pointPolygonTest(stomach_polygon, (hx, hy), False)
                    if inside >= 0:
                        # Check approximate depth proximity
                        stomach_depths = [left_shoulder[2], right_shoulder[2], left_hip[2], right_hip[2]]
                        avg_stomach_depth = np.mean(stomach_depths)
                        if abs(int(hz) - int(avg_stomach_depth)) < TOUCH_THRESHOLD:
                            contact_detected = True
                            cv2.circle(color_frame, (hx, hy), 12, (0, 255, 255), -1)
                            cv2.putText(color_frame, "HAND TOUCHING STOMACH!", (50,100),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                            break


            if contact_detected:
                cv2.putText(color_frame, "HAND TOUCHING BODY!", (50,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            # --- Visualization ---
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
