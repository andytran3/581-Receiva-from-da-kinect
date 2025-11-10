# mediapipe_receive.py (threaded version)
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

def recvall(sock, n):
    """Receive n bytes exactly."""
    buf = b''
    while len(buf) < n:
        data = sock.recv(n - len(buf))
        if not data:
            return None
        buf += data
    return buf

def receive_frames(conn):
    """Background thread: receive both color and depth frames."""
    global latest_color, latest_depth, running

    while running:
        # Read frame type
        frame_type = recvall(conn, 1)
        if not frame_type:
            running = False
            break

        # Read length
        raw_len = recvall(conn, 4)
        if not raw_len:
            running = False
            break
        length = struct.unpack(">L", raw_len)[0]

        # Read data
        data = recvall(conn, length)
        if not data:
            running = False
            break

        # Decode
        if frame_type == b'C':
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            with lock:
                latest_color = frame
        elif frame_type == b'D':
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
            with lock:
                latest_depth = frame

def main():
    global running

    # --- Setup MediaPipe ---
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=2,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)

    # --- Socket setup ---
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((HOST, PORT))
    sock.listen(1)
    print("Waiting for Kinect connection...")
    conn, addr = sock.accept()
    print(f"Connected to {addr}")

    # Start receiver thread
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

            # --- Pose detection ---
            results_pose = pose.process(frame_rgb)
            if results_pose.pose_landmarks:
                mp_drawing.draw_landmarks(
                    color_frame,
                    results_pose.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

            # --- Hand detection ---
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

            # --- Visualization ---
            depth_vis = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            combined = np.hstack((
                cv2.resize(color_frame, (640, 360)),
                cv2.resize(depth_vis, (640, 360))
            ))

            cv2.imshow("RGB + Depth + MediaPipe (Threaded)", combined)
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
