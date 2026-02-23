import cv2
import mediapipe as mp
import time

from detectors.face_recognizer import FaceRecognizer
from detectors.eye_detector import EyeDetector
from detectors.gaze_detector import GazeDetector
from utils.session_utils import generate_session_id
from utils.cloudinary_uploader import CloudinaryUploader
from utils.postgres_logger import PostgresLogger


class FrameProcessor:
    def __init__(self, expected_user=None, session_id=None):
        self.session_id = session_id or generate_session_id(prefix="video")
        self.expected_user = expected_user

        self.eye_detector = EyeDetector()
        self.gaze_detector = GazeDetector(h_dev=0.15, v_down=0.17, v_up=0.25, debug=False)
        self.face_recognizer = FaceRecognizer(tolerance=0.50)

        self.logger = PostgresLogger()

        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.6
        )

        self.frame_id = 0

        # Scheduling
        self.last_detection_time = 0
        self.last_gaze_time = 0
        self.last_identity_time = 0

        self.cached_face_boxes = []
        self.cached_identity = []
        self.cached_gaze = False
        self.cached_gaze_debug = {}

        self.no_face_counter = 0
        self.lookaway_counter = 0

        # Cloud upload cooldown
        self.last_upload_time = 0
        self.upload_cooldown = 4  # seconds

    # -------------------------------------------------
    def process_frame(self, frame):
        self.frame_id += 1
        anomalies = []

        if frame is None:
            return anomalies

        now = time.time()
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ---------------- FACE DETECTION ----------------
        if now - self.last_detection_time >= 1.0:
            results = self.face_detection.process(rgb_frame)
            face_boxes = []

            if results.detections:
                for d in results.detections:
                    bbox = d.location_data.relative_bounding_box

                    top = max(int(bbox.ymin * h), 0)
                    left = max(int(bbox.xmin * w), 0)
                    bottom = min(int((bbox.ymin + bbox.height) * h), h)
                    right = min(int((bbox.xmin + bbox.width) * w), w)

                    area = (bottom - top) * (right - left)
                    if area > 0.05 * w * h:
                        face_boxes.append((top, right, bottom, left))

            self.cached_face_boxes = face_boxes
            self.last_detection_time = now
        else:
            face_boxes = self.cached_face_boxes

        multiple_faces = len(face_boxes) > 1

        # ---------------- FACE RECOGNITION ----------------
        if now - self.last_identity_time >= 5.0:
            recognized_faces = self.face_recognizer.recognize(
                rgb_frame, face_boxes, expected_user=self.expected_user
            )
            self.cached_identity = recognized_faces
            self.last_identity_time = now
        else:
            recognized_faces = self.cached_identity

        if not recognized_faces:
            self.no_face_counter += 1
        else:
            self.no_face_counter = 0

        if self.no_face_counter >= 3:
            anomalies.append({"frame_id": self.frame_id, "reason": "no_face_detected"})

        if not recognized_faces:
            return anomalies

        # ---------------- GAZE ----------------
        face = recognized_faces[0]
        name = face["name"]
        top, right, bottom, left = face["location"]

        top, right, bottom, left = max(0, top), min(w, right), min(h, bottom), max(0, left)
        face_h, face_w = bottom - top, right - left

        if now - self.last_gaze_time >= 0.35:

            left_eye, right_eye, _ = self.eye_detector.get_eye_landmarks(rgb_frame)

            if left_eye and right_eye:
                left_center = self.eye_detector.compute_eye_center(left_eye)
                right_center = self.eye_detector.compute_eye_center(right_eye)

                left_center = (left_center[0] - left, left_center[1] - top)
                right_center = (right_center[0] - left, right_center[1] - top)

                looking_away, gaze_debug = self.gaze_detector.is_looking_away(
                    left_center, right_center, face_w, face_h
                )
            else:
                looking_away = False
                gaze_debug = {}

            self.cached_gaze = looking_away
            self.cached_gaze_debug = gaze_debug
            self.last_gaze_time = now
        else:
            looking_away = self.cached_gaze
            gaze_debug = self.cached_gaze_debug

        if looking_away:
            self.lookaway_counter += 1
        else:
            self.lookaway_counter = 0

        gaze_flag = self.lookaway_counter >= 1

        imposter_detected = name == "Unknown" or (
            self.expected_user and name != self.expected_user
        )

        # ---------------- DETERMINE ANOMALY ----------------
        anomaly_reason = None
        if gaze_flag:
            anomaly_reason = "looking_away"
        elif multiple_faces:
            anomaly_reason = "multiple_faces"
        elif imposter_detected:
            anomaly_reason = "imposter"
        elif self.no_face_counter >= 3:
            anomaly_reason = "no_face"

        cloud_url = None

        # ---------------- CLOUD + DB LOGGING ----------------
        if anomaly_reason and (now - self.last_upload_time) > self.upload_cooldown:

            cloud_url = CloudinaryUploader.upload_frame(
                frame,
                self.session_id,
                self.frame_id,
                anomaly_reason
            )

            self.logger.log_anomaly(
                frame_id=self.frame_id,
                name=name,
                expected_user=self.expected_user,
                face_visible=True,
                eyes_visible=True,
                looking_away=gaze_flag,
                multiple_faces=multiple_faces,
                imposter_detected=imposter_detected,
                session_id=self.session_id,
                frame_url=cloud_url
            )

            self.last_upload_time = now

        anomalies.append({
            "frame_id": self.frame_id,
            "name": name,
            "looking_away": gaze_flag,
            "multiple_faces": multiple_faces,
            "imposter_detected": imposter_detected,
            "frame_url": cloud_url,
            "gaze_debug": gaze_debug
        })

        return anomalies

    def close(self):
        self.face_detection.close()
        self.logger.close()