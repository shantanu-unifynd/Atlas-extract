# Atlas Extract — floor-plan extraction service.
# opencv-python-headless and rapidocr-onnxruntime bundle their own native
# libraries and OCR models, so the image needs no GUI stack and makes no
# network calls at runtime. libglib2.0-0 is the one shared lib OpenCV still
# links against on slim.
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 libxcb1 libsm6 libxext6 libxrender1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first so the layer caches across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 5600

# 0.0.0.0 so the port is reachable from outside the container.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5600"]
