---
title: Weapon Detection API
emoji: 🛡️
colorFrom: gray
colorTo: red
sdk: docker
pinned: false
---

# Weapon Detection API

Flask API serving a YOLO model trained to detect weapons in images.

## Endpoints

- `GET /` — health check
- `POST /predict` — send an image as form field `image`, returns JSON:
  ```json
  { "detections": [ { "label": "gun", "confidence": 0.91, "box": [x1, y1, x2, y2] } ] }
  ```

## Setup

1. Copy your trained `gun_detector_best.pt` weights into this folder (or set the `MODEL_PATH` env var to a different filename/path).
2. Push this folder to a Hugging Face Space (SDK: Docker, hardware: CPU basic).
3. Once live, copy the Space's URL + `/predict` into the frontend's API endpoint settings.
