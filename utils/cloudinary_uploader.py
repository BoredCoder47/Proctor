import cloudinary
import cloudinary.uploader
import os
import cv2
import traceback
from dotenv import load_dotenv


load_dotenv()

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)


class CloudinaryUploader:
    @staticmethod
    def upload_frame(frame, session_id, frame_id, anomaly_reason):
        if frame is None or frame.size == 0:
            return None

        try:
            # 🔥 Encode in memory (NO disk write)
            success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not success:
                return None

            upload = cloudinary.uploader.upload(
                buffer.tobytes(),
                folder=f"anomaly_logs/{session_id}",
                public_id=f"frame_{frame_id}_{anomaly_reason}",
                resource_type="image",
                overwrite=True
            )

            return upload.get("secure_url")

        except Exception as e:
            print("[CloudinaryUploader] Upload failed:", e)
            traceback.print_exc()
            return None