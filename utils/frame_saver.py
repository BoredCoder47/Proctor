import os
import cv2
import time


class FrameSaver:
    def __init__(self, base_dir="anomaly_frames"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def save(self, frame, session_id, reason, frame_id):
        session_dir = os.path.join(self.base_dir, session_id)
        os.makedirs(session_dir, exist_ok=True)

        ts = int(time.time() * 1000)
        filename = f"{reason}_f{frame_id}_{ts}.jpg"
        path = os.path.join(session_dir, filename)

        cv2.imwrite(path, frame)
        return path
