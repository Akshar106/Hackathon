"""
Compression microservice — FastAPI

Endpoints:
  POST /compress    — adaptive Huffman compress uploaded text or binary file
  POST /decompress  — decompress a previously compressed payload
  GET  /health      — liveness check

Start with:
    uvicorn compression_service.api:app --port 8001 --reload
"""

import io
import json
import time

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from compression_service.huffman import compress, decompress

app = FastAPI(title="Compression Service", version="1.0")


# ── Schemas ───────────────────────────────────────────────────────────────────

class DecompressRequest(BaseModel):
    bitstring: str
    original_length: int
    freqs: dict = {}     # not used by adaptive decoder; kept for API compatibility
    pad_bits: int = 0


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "algorithm": "adaptive_huffman"}


@app.post("/compress")
async def compress_file(file: UploadFile = File(...)):
    """
    Upload any file (text or binary).
    Returns compression metrics + the compressed payload.
    """
    data = await file.read()
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    t0 = time.perf_counter()
    payload = compress(data)
    latency_ms = (time.perf_counter() - t0) * 1000

    return {
        "filename": file.filename,
        "original_bytes": payload["stats"]["original_bytes"],
        "compressed_bytes": payload["stats"]["compressed_bytes"],
        "compression_ratio": payload["stats"]["compression_ratio"],
        "entropy_bits_per_symbol": payload["stats"]["entropy_bits_per_symbol"],
        "avg_bits_per_symbol": payload["stats"]["avg_bits_per_symbol"],
        "encoding_efficiency": payload["stats"]["encoding_efficiency"],
        "latency_ms": round(latency_ms, 2),
        # Payload fields needed for decompression
        "bitstring": payload["bitstring"],
        "packed_b64": payload["packed_b64"],
        "pad_bits": payload["pad_bits"],
        "original_length": payload["original_length"],
        "freqs": payload["freqs"],
    }


@app.post("/decompress")
async def decompress_payload(req: DecompressRequest):
    """
    Decompress a bitstring back to the original bytes.
    Returns the decoded text (UTF-8) or a hex dump for binary data.
    """
    t0 = time.perf_counter()
    try:
        recovered = decompress({
            "bitstring": req.bitstring,
            "original_length": req.original_length,
            "freqs": req.freqs,
        })
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Decompression failed: {exc}")
    latency_ms = (time.perf_counter() - t0) * 1000

    try:
        text = recovered.decode("utf-8")
        encoding = "utf-8"
    except UnicodeDecodeError:
        text = recovered.hex()
        encoding = "hex"

    return {
        "original_length": req.original_length,
        "recovered_length": len(recovered),
        "encoding": encoding,
        "text": text,
        "latency_ms": round(latency_ms, 2),
    }
