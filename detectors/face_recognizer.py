import os
import face_recognition
import numpy as np
import cv2


class FaceRecognizer:
    def __init__(self, known_faces_dir="known_faces", tolerance=0.50):
        """
        Strict face recognition for single-student proctoring.

        known_faces_dir:
            Folder containing reference face images (jpg/png/jpeg)

        tolerance:
            Lower = stricter matching (0.45–0.55 recommended for exams)
        """
        self.known_encodings = []
        self.known_names = []
        self.tolerance = tolerance

        self._load_known_faces(known_faces_dir)

    # -------------------------------------------------
    # Load known faces
    # -------------------------------------------------
    def _load_known_faces(self, known_faces_dir):
        if not os.path.exists(known_faces_dir):
            os.makedirs(known_faces_dir)
            print(f"[FaceRecognizer] Created directory: {known_faces_dir}")
            return

        print(f"[FaceRecognizer] Loading faces from '{known_faces_dir}'...")

        for filename in os.listdir(known_faces_dir):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            path = os.path.join(known_faces_dir, filename)

            try:
                image = face_recognition.load_image_file(path)
                encodings = face_recognition.face_encodings(image)

                if len(encodings) > 0:
                    self.known_encodings.append(encodings[0])
                    self.known_names.append(os.path.splitext(filename)[0])
                    print(f"[FaceRecognizer] Loaded: {filename}")
                else:
                    print(f"[FaceRecognizer] WARNING: No face found in {filename}")

            except Exception as e:
                print(f"[FaceRecognizer] Error loading {filename}: {e}")

        print(f"[FaceRecognizer] Total known faces: {len(self.known_names)}")

    # -------------------------------------------------
    # Normalize face boxes safely
    # -------------------------------------------------
    def _normalize_boxes(self, boxes, frame_shape):
        h, w = frame_shape[:2]
        normalized = []

        for box in boxes:
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue

            top, right, bottom, left = box

            top = max(0, min(int(top), h))
            right = max(0, min(int(right), w))
            bottom = max(0, min(int(bottom), h))
            left = max(0, min(int(left), w))

            if bottom > top and right > left:
                normalized.append((top, right, bottom, left))

        return normalized

    # -------------------------------------------------
    # Recognize faces
    # -------------------------------------------------
    def recognize(self, frame, face_boxes, expected_user=None):
        """
        frame MUST already be RGB.
        Returns list of:
        {
            "name": str,
            "location": (top, right, bottom, left),
            "expected_user": str
        }
        """
        results = []

        if frame is None or len(face_boxes) == 0:
            return results

        # Ensure RGB (avoid double conversion)
        if frame.shape[2] == 3 and frame.dtype == np.uint8:
            # Assume already RGB from FrameProcessor
            pass

        face_boxes = self._normalize_boxes(face_boxes, frame.shape)

        if not face_boxes:
            return results

        # Compute encodings
        try:
            encodings = face_recognition.face_encodings(
                frame,
                known_face_locations=face_boxes,
                model="small"  # faster, good enough
            )
        except Exception as e:
            print(f"[FaceRecognizer] Encoding error: {e}")
            return results

        if len(self.known_encodings) == 0:
            # No registered faces
            for (top, right, bottom, left) in face_boxes:
                results.append({
                    "name": "Unknown",
                    "location": (top, right, bottom, left),
                    "expected_user": expected_user
                })
            return results

        for encoding, (top, right, bottom, left) in zip(encodings, face_boxes):

            distances = face_recognition.face_distance(
                self.known_encodings,
                encoding
            )

            best_idx = np.argmin(distances)
            best_distance = distances[best_idx]

            if best_distance <= self.tolerance:
                name = self.known_names[best_idx]
            else:
                name = "Unknown"

            if expected_user:
                print(
                    f"[FaceRecognizer] Detected: {name} | "
                    f"Expected: {expected_user} | "
                    f"Distance: {best_distance:.4f}"
                )

            results.append({
                "name": name,
                "location": (top, right, bottom, left),
                "expected_user": expected_user
            })

        return results
