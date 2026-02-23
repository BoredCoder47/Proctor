import mediapipe as mp
import cv2
import numpy as np


class EyeDetector:
    def __init__(self, scale_factor=0.5):
        """
        Eye landmark detector using MediaPipe FaceMesh.

        scale_factor:
            Downscale factor for performance (0.4–0.6 recommended).
        """
        self.scale_factor = scale_factor

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            static_image_mode=False,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        self.LEFT_EYE_IDX = [33, 133, 160, 159, 158, 157, 173, 144, 145, 153, 154, 155]
        self.RIGHT_EYE_IDX = [362, 263, 387, 386, 385, 384, 398, 373, 374, 380, 381, 382]

    # -------------------------------------------------
    # Get eye landmarks
    # -------------------------------------------------
    def get_eye_landmarks(self, frame):
        if frame is None or frame.size == 0:
            return None, None, None

        h, w, _ = frame.shape

        # Downscale for speed
        small_frame = cv2.resize(
            frame,
            (0, 0),
            fx=self.scale_factor,
            fy=self.scale_factor
        )

        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb_small)

        if not results.multi_face_landmarks:
            return None, None, None

        face_landmarks = results.multi_face_landmarks[0]

        # Get scaled width/height
        small_h, small_w, _ = small_frame.shape

        # Convert normalized coords → small image coords → original coords
        def scale_point(landmark):
            x_small = landmark.x * small_w
            y_small = landmark.y * small_h

            x_orig = int(x_small / self.scale_factor)
            y_orig = int(y_small / self.scale_factor)

            return (x_orig, y_orig)

        left_eye = [scale_point(face_landmarks.landmark[i]) for i in self.LEFT_EYE_IDX]
        right_eye = [scale_point(face_landmarks.landmark[i]) for i in self.RIGHT_EYE_IDX]

        return left_eye, right_eye, face_landmarks

    # -------------------------------------------------
    # Compute center
    # -------------------------------------------------
    @staticmethod
    def compute_eye_center(eye_landmarks):
        if not eye_landmarks:
            return None

        pts = np.array(eye_landmarks, dtype=np.float32)
        center = np.mean(pts, axis=0)

        return (float(center[0]), float(center[1]))

    # -------------------------------------------------
    # Cleanup
    # -------------------------------------------------
    def close(self):
        self.face_mesh.close()
