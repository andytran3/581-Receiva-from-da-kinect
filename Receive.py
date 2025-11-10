# mediapipe_receive_touch_fixed.py
import socket
import cv2
import numpy as np
import struct
import mediapipe as mp
import threading

HOST = '127.0.0.1'
PORT = 5001

# Shared frames and lock
latest_color = None
latest_depth = None
lock = threading.Lock()
running = True

TOUCH_THRESHOLD = 25  # adjust based on depth units

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
    recv_thread.start()

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
            contact_detected = False
            for hx, hy, hz in hand_joints:
                for bx, by, bz in body_joints:
                    dx = hx - bx
                    dy = hy - by
                    dz = int(hz) - int(bz)
                    dist = np.sqrt(dx**2 + dy**2 + dz**2)
                    if dist < TOUCH_THRESHOLD:
                        contact_detected = True
                        cv2.circle(color_frame, (hx, hy), 10, (0,0,255), -1)
                        break
                if contact_detected:
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
