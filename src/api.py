"""FastAPI serving for HAM10000 inference.

Endpoints:
  GET  /health   -> model status
  POST /predict  -> multipart image upload, returns predicted class + probabilities
"""
import os
import tempfile
from functools import lru_cache

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .model import CLASS_NAMES, MODEL_PATH, load_model, predict_image

app = FastAPI(title="Skin Lesion Classifier (HAM10000, ConvNeXt-Base)")


@lru_cache(maxsize=1)
def get_model():
    ckpt = os.getenv("MODEL_PATH", MODEL_PATH)
    return load_model(ckpt)


@app.get("/health")
def health():
    ckpt = os.getenv("MODEL_PATH", MODEL_PATH)
    return {
        "status": "ok",
        "model_path": ckpt,
        "model_loaded": os.path.exists(ckpt),
        "classes": CLASS_NAMES,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must be an image file")
    try:
        model = get_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    suffix = os.path.splitext(file.filename or "upload.jpg")[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        result = predict_image(tmp_path, model)
    except Exception as exc:  # corrupt image etc.
        raise HTTPException(status_code=400, detail=f"Inference failed: {exc}")
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    return JSONResponse(result)
