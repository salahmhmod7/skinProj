# Inference runtime for HAM10000 skin-lesion classifier.
# Default: CPU-only PyTorch (works everywhere). For NVIDIA GPU, see notes below.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# opencv-headless still needs libgl/libglib at runtime for some builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src/ ./src/

# Model + data are mounted at runtime, NOT baked into the image
# (keeps the image small; archive.zip / HAM10000 images are GBs).
# Expected mount: -v ./data:/app/data:ro
ENV MODEL_PATH=/app/data/convnext_clinical_best.pth \
    IMG_SIZE=384 \
    MEL_THRESHOLD=0.30

EXPOSE 8000

# Default: serve the FastAPI app. Override for CLI inference, e.g.:
#   docker run --rm -v ./data:/app/data:ro skinproj python -m src.infer --image /app/data/HAM10000_images_part_1/ISIC_0027419.jpg
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

# --- GPU variant (uncomment / build separately) ---
# FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
# ... same steps, but drop the --extra-index-url CPU line and install CUDA wheels ...
