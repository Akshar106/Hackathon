"""
OCR microservice — FastAPI

Endpoints:
  POST /ocr          — run full OCR pipeline on an uploaded image
  POST /ocr/denoise  — return the denoised image (JPEG) for inspection
  GET  /health       — liveness check

Start with:
    uvicorn ocr_service.api:app --port 8000 --reload
"""

import io
import os
import time
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from PIL import Image

from ocr_service.predict import load_classifier, predict_image
from ocr_service.preprocess import load_denoiser, denoise_array, load_grayscale

# ── Model paths ───────────────────────────────────────────────────────────────

_BASE = Path(__file__).resolve().parent.parent
CLASSIFIER_WEIGHTS = str(_BASE / "models" / "char_classifier.pth")
DENOISER_WEIGHTS   = str(_BASE / "models" / "denoiser.pth")

DEVICE = os.environ.get("OCR_DEVICE", "cpu")

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="OCR Service", version="1.0")


@app.on_event("startup")
async def _load_models():
    if not Path(CLASSIFIER_WEIGHTS).exists():
        raise RuntimeError(f"Classifier weights not found: {CLASSIFIER_WEIGHTS}")
    load_classifier(CLASSIFIER_WEIGHTS, device=DEVICE)

    if Path(DENOISER_WEIGHTS).exists():
        load_denoiser(DENOISER_WEIGHTS, device=DEVICE)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "device": DEVICE}


@app.post("/ocr")
async def ocr(
    file: UploadFile = File(...),
    denoise: bool = True,
):
    """
    Upload a scanned document image (PNG/JPEG/TIFF).
    Returns OCR text with latency breakdown.
    """
    if file.content_type not in ("image/png", "image/jpeg", "image/tiff", "image/bmp"):
        raise HTTPException(status_code=415, detail="Unsupported image type")

    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot open image: {exc}")

    t0 = time.perf_counter()
    result = predict_image(image, denoise=denoise)
    latency_ms = (time.perf_counter() - t0) * 1000

    return {
        "text": result["text"],
        "characters": result["characters"],
        "num_chars": result["num_chars"],
        "denoise_enabled": denoise,
        "latency_ms": round(latency_ms, 2),
    }


@app.post("/ocr/denoise")
async def ocr_denoise(file: UploadFile = File(...)):
    """
    Return the denoised version of the uploaded image for visual inspection.
    Response is a grayscale PNG.
    """
    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot open image: {exc}")

    gray = load_grayscale(image)
    denoised = denoise_array(gray)

    out_img = Image.fromarray(denoised, mode="L")
    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    buf.seek(0)
    return Response(content=buf.read(), media_type="image/png")
