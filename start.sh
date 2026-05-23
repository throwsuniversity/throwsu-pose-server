#!/bin/bash
export RTMLIB_CACHE=/workspace/.rtmlib

pip install "numpy<2" onnxruntime-gpu==1.19.2 fastapi uvicorn httpx \
  "opencv-python-headless==4.8.1.78" "opencv-python==4.8.1.78" \
  "opencv-contrib-python==4.8.1.78" rtmlib --quiet

cd /workspace && uvicorn main:app --host 0.0.0.0 --port 8000
