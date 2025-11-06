# mediapipe_receive.py
import socket
import cv2
import numpy as np
import struct
import mediapipe as mp

HOST = '127.0.0.1'
PORT = 5001

def receive_frame(sock):
    # Receive length
    raw_len = recvall(sock, 4)
    if not raw_len:
        return None
    length = struct.unpack(">L", raw_len)[0]
    # Receive frame data
    data = recvall(sock, length)
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    return frame

def recvall(sock, n):
    """Receive n bytes exactly"""
    buf = b''
    while len(buf) < n:
        data = sock.recv(n - len(buf))
        if not data:
            return None
        buf += data
    return buf

def main():
    # Initialize MediaPipe Pose and Hands
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
    
    try:
        while True:
            frame = receive_frame(conn)
            if frame is None:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # --- Pose detection ---
            results_pose = pose.process(frame_rgb)

            if results_pose.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
                )

            # --- Hand detection ---
            results_hands = hands.process(frame_rgb)
            if results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0,255,255), thickness=2)
                    )

            # Show combined result
            cv2.imshow("MediaPipe Pose + Hands", cv2.resize(frame, (960, 540)))
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        conn.close()
        sock.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
