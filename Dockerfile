FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=10000 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

WORKDIR /app

# OpenCV (used internally by ultralytics) needs these system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 10000

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} --workers 1 app:app"]
