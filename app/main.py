import os
import time
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from workers.frame_processor import FrameProcessor
from workers.audio_processor import AudioProcessor


# -------------------------------------------------
# Load environment
# -------------------------------------------------
load_dotenv()

# -------------------------------------------------
# FastAPI App
# -------------------------------------------------
app = FastAPI(title="Local Exam Proctoring API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # LAN testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Initialize processors
# -------------------------------------------------
shared_session_id = f"session_{int(time.time())}"

video_processor = FrameProcessor(
    expected_user="photo1",
    session_id=shared_session_id
)

audio_processor = AudioProcessor()

print("Session ID:", shared_session_id)

# -------------------------------------------------
# Health
# -------------------------------------------------
@app.get("/")
def health():
    return {"status": "running", "session_id": shared_session_id}

# -------------------------------------------------
# VIDEO ENDPOINT
# -------------------------------------------------
@app.post("/process-frame")
async def process_frame(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        if not contents:
            return {"error": "Empty frame"}

        np_arr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return {"error": "Invalid frame"}

        frame = cv2.resize(frame, (640, 480))

        anomalies = video_processor.process_frame(frame)

        return {
            "session_id": shared_session_id,
            "anomalies": anomalies
        }

    except Exception as e:
        return {"error": str(e)}

# -------------------------------------------------
# AUDIO ENDPOINT (PCM STREAM)
# -------------------------------------------------
@app.post("/process-audio")
async def process_audio(request: Request):
    try:
        raw = await request.body()

        if not raw:
            return {"noise": False}

        audio = np.frombuffer(raw, dtype=np.float32)

        if len(audio) == 0:
            return {"noise": False}

        return audio_processor.process_pcm(audio)

    except Exception as e:
        print("Audio error:", e)
        return {"noise": False}

# -------------------------------------------------
# Shutdown
# -------------------------------------------------
@app.on_event("shutdown")
def shutdown_event():
    try:
        video_processor.close()
        print("Processors cleaned.")
    except:
        pass