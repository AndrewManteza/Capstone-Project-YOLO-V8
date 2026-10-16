FROM python:3.11-slim

WORKDIR /app

# OpenCV (used internally by ultralytics) needs these system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch: far smaller and faster to build than the default CUDA build
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860
ENV PORT=7860
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics

CMD ["python", "app.py"]
