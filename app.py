import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)  # allows your GitHub Pages frontend (a different domain) to call this API

# Path to your trained weights. Copy your best.pt into this folder,
# or set the MODEL_PATH environment variable to point elsewhere.
MODEL_PATH = os.environ.get("MODEL_PATH", "best.pt")
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", 0.4))

model = YOLO(MODEL_PATH)


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Weapon detection API is running."})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Send it as form field 'image'."}), 400

    file = request.files["image"]
    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "Could not read the uploaded file as an image."}), 400

    results = model.predict(image, conf=CONFIDENCE_THRESHOLD, iou=0.4, verbose=False)
    result = results[0]
    names = result.names  # dict of class id -> class name, from your dataset/classes.txt

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        label = names[cls_id] if isinstance(names, dict) else names[cls_id]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        detections.append({
            "label": label,
            "confidence": round(confidence, 4),
            "box": [x1, y1, x2, y2],
        })

    return jsonify({"detections": detections})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
