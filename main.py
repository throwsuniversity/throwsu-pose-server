import os
os.environ['RTMLIB_CACHE'] = '/workspace/.rtmlib'

import asyncio
import json
import httpx
import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from rtmlib import Wholebody

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading pose model...")
pose_model = Wholebody(
    mode='performance',
    backend='onnxruntime',
    device='cuda'
)
print("Pose model loaded.")


class AnalyzeRequest(BaseModel):
    video_url: str
    fps: Optional[float] = None


@app.get("/health")
def health():
    return {"status": "ok", "model": "rtmw-l"}


@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    async def generate():
        # Download video
        tmp_path = "/tmp/input_video.mp4"
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.get(request.video_url)
            with open(tmp_path, "wb") as f:
                f.write(r.content)

        cap = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = request.fps or cap.get(cv2.CAP_PROP_FPS) or 60
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames_data = []
        frame_index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            keypoints, scores = pose_model(frame)

            landmarks = []
            landmarks_raw = []
            if keypoints is not None and len(keypoints) > 0:
                kp = keypoints[0]
                sc = scores[0] if scores is not None else [1.0] * len(kp)
                for i, (point, score) in enumerate(zip(kp, sc)):
                    lm = {
                        "x": float(point[0]) / width,
                        "y": float(point[1]) / height,
                        "z": 0.0,
                        "visibility": float(score)
                    }
                    landmarks.append(lm)
                    landmarks_raw.append(lm)

            t = frame_index / video_fps
            frames_data.append({
                "frameIndex": frame_index,
                "t": t,
                "landmarks": landmarks,
                "landmarks_raw": landmarks_raw,
                "imageWidth": width,
                "imageHeight": height
            })

            frame_index += 1
            progress = int((frame_index / max(total_frames, 1)) * 100)
            msg = json.dumps({
                "type": "progress",
                "progress": progress,
                "frame": frame_index,
                "total": total_frames
            })
            yield f"data: {msg}\n\n"
            await asyncio.sleep(0)

        cap.release()

        result = {
            "type": "complete",
            "frames": frames_data,
            "fps": video_fps,
            "frameCount": frame_index
        }
        yield f"data: {json.dumps(result)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
