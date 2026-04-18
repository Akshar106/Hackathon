"""
End-to-end pipeline orchestrator.

Runs the full pipeline on a document image:
  1. OCR microservice  — denoise + segment + classify characters
  2. Compression service — adaptive Huffman encode the OCR text

Also supports a benchmarking mode that measures per-stage latency.

Usage:
    # Single image, services already running
    python pipeline.py --image path/to/scan.png

    # Direct mode (no running services needed — loads models in-process)
    python pipeline.py --image path/to/scan.png --direct

    # Benchmark on a directory of images
    python pipeline.py --benchmark --image-dir path/to/images/ --direct
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── Service URLs ──────────────────────────────────────────────────────────────

OCR_URL         = os.environ.get("OCR_URL",         "http://localhost:8000")
COMPRESS_URL    = os.environ.get("COMPRESS_URL",    "http://localhost:8001")


# ── HTTP-based pipeline ───────────────────────────────────────────────────────

def run_via_services(image_path: str, denoise: bool = True) -> dict:
    """Call both microservices over HTTP."""
    import requests

    # Detect MIME type from extension
    ext = Path(image_path).suffix.lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "tiff": "image/tiff", "bmp": "image/bmp"}.get(ext.lstrip("."), "image/png")

    # Stage 1: OCR
    t0 = time.perf_counter()
    with open(image_path, "rb") as f:
        ocr_resp = requests.post(
            f"{OCR_URL}/ocr",
            files={"file": (Path(image_path).name, f, mime)},
            params={"denoise": denoise},
            timeout=120,
        )
    ocr_resp.raise_for_status()
    ocr_result = ocr_resp.json()
    ocr_latency = (time.perf_counter() - t0) * 1000

    ocr_text = ocr_result["text"]

    # Stage 2: Compression
    t1 = time.perf_counter()
    compress_resp = requests.post(
        f"{COMPRESS_URL}/compress",
        files={"file": ("ocr_text.txt", ocr_text.encode("utf-8"), "text/plain")},
        timeout=120,
    )
    compress_resp.raise_for_status()
    compress_result = compress_resp.json()
    compress_latency = (time.perf_counter() - t1) * 1000

    total_latency = (time.perf_counter() - t0) * 1000

    return {
        "image": image_path,
        "ocr_text": ocr_text,
        "num_chars": ocr_result["num_chars"],
        "ocr_latency_ms":      round(ocr_latency, 2),
        "compress_latency_ms": round(compress_latency, 2),
        "total_latency_ms":    round(total_latency, 2),
        "compression": {
            "original_bytes":          compress_result["original_bytes"],
            "compressed_bytes":        compress_result["compressed_bytes"],
            "compression_ratio":       compress_result["compression_ratio"],
            "entropy_bits_per_symbol": compress_result["entropy_bits_per_symbol"],
            "avg_bits_per_symbol":     compress_result["avg_bits_per_symbol"],
            "encoding_efficiency":     compress_result["encoding_efficiency"],
        },
    }


# ── Direct (in-process) pipeline ─────────────────────────────────────────────

def _ensure_models_loaded(device: str = "cpu"):
    from ocr_service.predict    import load_classifier, _classifier
    from ocr_service.preprocess import load_denoiser,   _denoiser

    base = Path(__file__).parent
    clf_path      = base / "models" / "char_classifier.pth"
    denoiser_path = base / "models" / "denoiser.pth"

    if _classifier is None:
        if not clf_path.exists():
            raise FileNotFoundError(f"Classifier weights not found: {clf_path}")
        load_classifier(str(clf_path), device=device)

    if _denoiser is None and denoiser_path.exists():
        load_denoiser(str(denoiser_path), device=device)


def run_direct(image_path: str, denoise: bool = True, device: str = "cpu") -> dict:
    """Run both stages in-process without HTTP overhead."""
    from ocr_service.predict import predict_image
    from compression_service.huffman import compress as huff_compress

    _ensure_models_loaded(device)

    # Stage 1: OCR
    t0 = time.perf_counter()
    ocr_result = predict_image(image_path, denoise=denoise)
    ocr_latency = (time.perf_counter() - t0) * 1000

    ocr_text = ocr_result["text"]

    # Stage 2: Compression
    t1 = time.perf_counter()
    payload = huff_compress(ocr_text.encode("utf-8"))
    compress_latency = (time.perf_counter() - t1) * 1000

    total_latency = (time.perf_counter() - t0) * 1000
    stats = payload["stats"]

    return {
        "image": image_path,
        "ocr_text": ocr_text,
        "num_chars": ocr_result["num_chars"],
        "ocr_latency_ms":      round(ocr_latency, 2),
        "compress_latency_ms": round(compress_latency, 2),
        "total_latency_ms":    round(total_latency, 2),
        "compression": {
            "original_bytes":          stats["original_bytes"],
            "compressed_bytes":        stats["compressed_bytes"],
            "compression_ratio":       stats["compression_ratio"],
            "entropy_bits_per_symbol": stats["entropy_bits_per_symbol"],
            "avg_bits_per_symbol":     stats["avg_bits_per_symbol"],
            "encoding_efficiency":     stats["encoding_efficiency"],
        },
    }


# ── Benchmarking ──────────────────────────────────────────────────────────────

def benchmark(image_dir: str, direct: bool = True, device: str = "cpu") -> dict:
    """Run pipeline on all PNG/JPEG images in a directory and aggregate metrics."""
    image_paths = sorted(
        list(Path(image_dir).glob("*.png"))
        + list(Path(image_dir).glob("*.jpg"))
        + list(Path(image_dir).glob("*.jpeg"))
    )

    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    results = []
    for p in image_paths:
        try:
            if direct:
                r = run_direct(str(p), device=device)
            else:
                r = run_via_services(str(p))
            results.append(r)
            print(f"  {p.name}: {r['num_chars']} chars | "
                  f"ratio={r['compression']['compression_ratio']} | "
                  f"total={r['total_latency_ms']:.1f}ms")
        except Exception as exc:
            print(f"  {p.name}: ERROR — {exc}")

    if not results:
        return {"error": "All images failed"}

    def _avg(key):
        return round(sum(r[key] for r in results) / len(results), 2)

    def _avg_nested(outer, inner):
        return round(sum(r[outer][inner] for r in results) / len(results), 4)

    return {
        "num_images": len(results),
        "avg_ocr_latency_ms":      _avg("ocr_latency_ms"),
        "avg_compress_latency_ms": _avg("compress_latency_ms"),
        "avg_total_latency_ms":    _avg("total_latency_ms"),
        "avg_compression_ratio":   _avg_nested("compression", "compression_ratio"),
        "avg_entropy":             _avg_nested("compression", "entropy_bits_per_symbol"),
        "avg_encoding_efficiency": _avg_nested("compression", "encoding_efficiency"),
        "per_image": results,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="OCR + Compression pipeline")
    p.add_argument("--image",     type=str, help="Path to a single input image")
    p.add_argument("--image-dir", type=str, help="Directory of images (for --benchmark)")
    p.add_argument("--direct",    action="store_true",
                   help="Run in-process (no running services needed)")
    p.add_argument("--benchmark", action="store_true",
                   help="Benchmark on all images in --image-dir")
    p.add_argument("--no-denoise", action="store_true",
                   help="Skip denoising stage")
    p.add_argument("--device",    type=str, default="cpu",
                   help="Torch device for direct mode (cpu / cuda / mps)")
    p.add_argument("--output",    type=str, default="",
                   help="Write JSON result to this path")
    return p.parse_args()


def main():
    args = parse_args()
    denoise = not args.no_denoise

    if args.benchmark:
        if not args.image_dir:
            print("ERROR: --benchmark requires --image-dir")
            sys.exit(1)
        print(f"Benchmarking on {args.image_dir} ...")
        result = benchmark(args.image_dir, direct=args.direct, device=args.device)
    elif args.image:
        print(f"Processing {args.image} ...")
        if args.direct:
            result = run_direct(args.image, denoise=denoise, device=args.device)
        else:
            result = run_via_services(args.image, denoise=denoise)
    else:
        print("ERROR: specify --image or --benchmark --image-dir")
        sys.exit(1)

    output = json.dumps(result, indent=2)
    print(output)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"\nResult saved to {args.output}")


if __name__ == "__main__":
    main()
