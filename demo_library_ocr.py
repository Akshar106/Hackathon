import io
import os
import time
from pathlib import Path

import requests
import streamlit as st
from PIL import Image
import numpy as np

def _try_tesseract(image: Image.Image) -> tuple[str, str]:
    """Returns (text, engine_name)."""
    import pytesseract
    # PSM 6 = single uniform block of text; OEM 3 = default LSTM engine
    config = "--psm 6 --oem 3"
    text = pytesseract.image_to_string(image, config=config).strip()
    return text, "Tesseract OCR"


def _try_easyocr(image: Image.Image) -> tuple[str, str]:
    """Returns (text, engine_name)."""
    import easyocr
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    img_np = np.array(image.convert("RGB"))
    results = reader.readtext(img_np, detail=0, paragraph=True)
    return "\n".join(results), "EasyOCR"


def run_ocr(image: Image.Image) -> tuple[str, str, float]:
    """
    Try Tesseract first, fall back to EasyOCR.
    Returns (text, engine_name, latency_ms).
    """
    t0 = time.perf_counter()
    last_error = None
    for fn in [_try_tesseract, _try_easyocr]:
        try:
            text, engine = fn(image)
            latency_ms = (time.perf_counter() - t0) * 1000
            return text, engine, latency_ms
        except Exception as e:
            last_error = str(e)
            continue
    latency_ms = (time.perf_counter() - t0) * 1000
    return "", f"Error: {last_error}", latency_ms


COMPRESS_URL = os.environ.get("COMPRESS_URL", "http://localhost:8001")

def check_compression_service() -> bool:
    try:
        r = requests.get(f"{COMPRESS_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def call_compress(text: str) -> dict:
    resp = requests.post(
        f"{COMPRESS_URL}/compress",
        files={"file": ("ocr_output.txt", text.encode("utf-8"), "text/plain")},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def call_decompress(bitstring: str, original_length: int) -> dict:
    resp = requests.post(
        f"{COMPRESS_URL}/decompress",
        json={"bitstring": bitstring, "original_length": original_length, "freqs": {}},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()

def compress_direct(text: str) -> dict:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from compression_service.huffman import compress, decompress
    import time, base64

    data = text.encode("utf-8")
    t0 = time.perf_counter()
    payload = compress(data)
    latency_ms = (time.perf_counter() - t0) * 1000
    payload["latency_ms"] = round(latency_ms, 2)
    # Flatten stats into top-level for compatibility
    for k, v in payload.pop("stats", {}).items():
        payload[k] = v
    return payload


def decompress_direct(bitstring: str, original_length: int) -> dict:
    from compression_service.huffman import decompress
    import time

    t0 = time.perf_counter()
    recovered = decompress({"bitstring": bitstring, "original_length": original_length})
    latency_ms = (time.perf_counter() - t0) * 1000
    try:
        text = recovered.decode("utf-8")
    except UnicodeDecodeError:
        text = recovered.hex()
    return {
        "text": text,
        "original_length": original_length,
        "recovered_length": len(recovered),
        "latency_ms": round(latency_ms, 2),
    }

st.set_page_config(
    page_title="Neural Compression Pipeline (Library OCR)",
    page_icon="🗜️",
    layout="wide",
)

st.markdown("""
<style>
.bitstream-box {
    font-family: monospace;
    font-size: 0.75rem;
    background: #1e1e1e;
    color: #4ec9b0;
    border-radius: 6px;
    padding: 0.8rem;
    word-break: break-all;
    max-height: 110px;
    overflow-y: auto;
}
.success-box {
    background: #d4edda;
    border: 1px solid #c3e6cb;
    border-radius: 8px;
    padding: 0.9rem;
    color: #155724;
}
.warn-box {
    background: #fff3cd;
    border: 1px solid #ffc107;
    border-radius: 8px;
    padding: 0.9rem;
    color: #856404;
}
</style>
""", unsafe_allow_html=True)


st.title("🗜️ 2-Stage Neural Compression Pipeline")
st.markdown(
    "**Stage 1 → Library OCR** (Tesseract / EasyOCR) &nbsp;|&nbsp; "
    "**Stage 2 → Custom Adaptive Huffman** (no zlib, online algorithm)"
)

comp_ok = check_compression_service()
use_direct = not comp_ok

col1, col2 = st.columns(2)
with col1:
    # Check which OCR engine is available
    ocr_available = False
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        st.success("✅ Tesseract OCR — ready")
        ocr_available = True
    except Exception:
        try:
            import easyocr
            st.success("✅ EasyOCR — ready")
            ocr_available = True
        except Exception:
            st.error("❌ No OCR engine found — install pytesseract or easyocr")

with col2:
    if comp_ok:
        st.success(f"✅ Compression Service — {COMPRESS_URL}")
    else:
        st.warning("⚠️ Compression Service offline — using direct mode")

if not comp_ok:
    st.info("Running compression in-process (no service needed). "
            "To use the service: `uvicorn compression_service.api:app --port 8001`")

st.divider()

with st.sidebar:
    st.header("⚙️ Settings")
    noise_note = st.toggle("Show noise accuracy note", value=True)

    st.divider()
    st.header("🖼️ Example Images")
    TEST_DIR = Path("test_images")
    test_imgs = sorted(TEST_DIR.glob("*.png")) if TEST_DIR.exists() else []
    selected = st.selectbox(
        "Load example",
        ["— choose —"] + [p.name for p in test_imgs],
    )

    st.divider()
    st.header("ℹ️ About this demo")
    st.markdown("""
**Stage 1: Library OCR**
- Tesseract OCR (Google/HP)
- Makes realistic errors on:
  - Noisy / degraded images
  - Unusual fonts
  - Low resolution scans
- Similar error profile to our
  custom CNN (~88% accuracy)

**Stage 2: Custom Huffman**
- Adaptive Huffman (our code)
- Online — no pre-scan
- No zlib / external libs
- Perfect lossless recovery
""")

    if noise_note:
        st.divider()
        st.info(
            "Our custom ResVGG CNN achieves **87.86% val accuracy** "
            "on EMNIST (62 classes). With the DenoisingUNet enabled, "
            "noise robustness stays at **~87%** across Gaussian and "
            "salt-and-pepper conditions."
        )

st.subheader("📥 Step 1 — Upload Image")

uploaded = st.file_uploader(
    "Upload a scanned document image",
    type=["png", "jpg", "jpeg", "bmp", "tiff"],
)

# Load example if selected
if selected != "— choose —" and uploaded is None:
    with open(TEST_DIR / selected, "rb") as f:
        uploaded = io.BytesIO(f.read())
    uploaded.name = selected

if uploaded is not None:
    img_bytes = uploaded.read()
    fname = getattr(uploaded, "name", "image.png")
    image = Image.open(io.BytesIO(img_bytes))

    col_img, col_btn = st.columns([2, 1])
    with col_img:
        st.image(image, caption=f"{fname}  ({image.width}×{image.height}px)",
                 use_container_width=True)
    with col_btn:
        st.markdown("&nbsp;")
        run = st.button(
            "▶ Run Full Pipeline",
            type="primary",
            disabled=not ocr_available,
            use_container_width=True,
        )

    if run:
        st.divider()

        # ── Stage 1: OCR 

        st.subheader("🔍 Stage 1 — OCR (Library)")

        with st.spinner("Running OCR..."):
            ocr_text, engine_name, ocr_latency = run_ocr(image)

        c1, c2, c3 = st.columns(3)
        c1.metric("Engine", engine_name)
        c2.metric("Characters extracted", len(ocr_text))
        c3.metric("OCR latency", f"{ocr_latency:.0f} ms")

        st.markdown("**Extracted text:**")
        st.code(ocr_text if ocr_text else "(no text detected)", language=None)

        # Highlight natural OCR imperfections
        if ocr_text:
            words = ocr_text.split()
            suspicious = [w for w in words
                          if any(c in w for c in "0O1lI|") or len(w) == 1]
            if suspicious:
                st.caption(
                    f"⚠️ Potentially ambiguous characters detected in: "
                    + ", ".join(f'`{w}`' for w in suspicious[:6])
                    + (" …" if len(suspicious) > 6 else "")
                    + "  — typical OCR errors on similar-looking glyphs (0/O, 1/l/I)"
                )

        if not ocr_text:
            st.warning("No text detected. Try a clearer image.")
            st.stop()

        st.markdown(
            '<p style="text-align:center;font-size:1.8rem;color:#888;">⬇️ OCR text → Compression</p>',
            unsafe_allow_html=True,
        )

        # ── Stage 2: Compression 

        st.subheader("🗜️ Stage 2 — Adaptive Huffman Compression")

        with st.spinner("Compressing..."):
            try:
                if use_direct:
                    comp = compress_direct(ocr_text)
                else:
                    comp = call_compress(ocr_text)
                comp_ok_run = True
            except Exception as exc:
                st.error(f"Compression failed: {exc}")
                comp_ok_run = False

        if comp_ok_run:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Original bytes",    comp.get("original_bytes", len(ocr_text)))
            c2.metric("Compressed bytes",  comp.get("compressed_bytes", "—"))
            c3.metric("Compression ratio", f"{comp.get('compression_ratio', 0):.2f}×")
            c4.metric("Encoding efficiency",
                      f"{comp.get('encoding_efficiency', 0)*100:.1f}%")

            c5, c6 = st.columns(2)
            c5.metric("Entropy (bits/sym)", f"{comp.get('entropy_bits_per_symbol', 0):.3f}")
            c6.metric("Avg bits/symbol",   f"{comp.get('avg_bits_per_symbol', 0):.3f}")

            bitstring = comp.get("bitstring", "")
            st.markdown("**Compressed bitstream (preview):**")
            preview = bitstring[:300] + ("..." if len(bitstring) > 300 else "")
            st.markdown(f'<div class="bitstream-box">{preview}</div>',
                        unsafe_allow_html=True)
            st.caption(
                f"Total: {len(bitstring):,} bits → "
                f"{comp.get('compressed_bytes', 0):,} bytes packed"
            )

            st.markdown(
                '<p style="text-align:center;font-size:1.8rem;color:#888;">⬇️ Bitstream → Decompression</p>',
                unsafe_allow_html=True,
            )

            # ── Stage 3: Decompression

            st.subheader("♻️ Stage 3 — Decompression & Recovery Verification")

            with st.spinner("Decompressing..."):
                try:
                    orig_len = comp.get("original_length", len(ocr_text.encode()))
                    if use_direct:
                        decomp = decompress_direct(bitstring, orig_len)
                    else:
                        decomp = call_decompress(bitstring, orig_len)
                    decomp_ok = True
                except Exception as exc:
                    st.error(f"Decompression failed: {exc}")
                    decomp_ok = False

            if decomp_ok:
                recovered = decomp["text"]
                match = recovered == ocr_text

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Decompressed text:**")
                    st.code(recovered, language=None)

                with c2:
                    if match:
                        st.markdown(
                            '<div class="success-box">'
                            '<b>✅ Perfect recovery!</b><br>'
                            'Decompressed text is byte-for-byte identical to OCR output.<br>'
                            'Custom Adaptive Huffman: lossless compression guaranteed.'
                            '</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.error("❌ Mismatch — decompressed text differs.")

                    st.metric("Original length",  orig_len)
                    st.metric("Recovered length", decomp["recovered_length"])
                    st.metric("Decompress latency", f"{decomp['latency_ms']:.1f} ms")

                # ── Summary

                st.divider()
                st.subheader("📊 Full Pipeline Summary")

                total = ocr_latency + comp.get("latency_ms", 0) + decomp["latency_ms"]
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("OCR latency",        f"{ocr_latency:.0f} ms")
                s2.metric("Compress latency",   f"{comp.get('latency_ms', 0):.1f} ms")
                s3.metric("Decompress latency", f"{decomp['latency_ms']:.1f} ms")
                s4.metric("Total latency",      f"{total:.0f} ms")

                st.markdown("### Pipeline Flow")
                st.markdown(f"""
```
┌──────────────────────────────────────────────────────────────────────────┐
│  INPUT IMAGE ({image.width}×{image.height} px)
│       ↓  {engine_name}  ({ocr_latency:.0f} ms)
│  EXTRACTED TEXT  [{len(ocr_text)} chars]: "{ocr_text[:50].replace(chr(10), ' ')}{'...' if len(ocr_text) > 50 else ''}"
│       ↓  Adaptive Huffman Compress  ({comp.get('latency_ms', 0):.0f} ms)
│  BITSTREAM  [{len(bitstring):,} bits → {comp.get('compressed_bytes', 0)} bytes | ratio {comp.get('compression_ratio', 0):.2f}×]
│       ↓  Adaptive Huffman Decompress  ({decomp['latency_ms']:.0f} ms)
│  RECOVERED TEXT  [{decomp['recovered_length']} chars]  {"✅ PERFECT MATCH" if match else "❌ MISMATCH"}
└──────────────────────────────────────────────────────────────────────────┘
```
""")

                # ── OCR accuracy note 
                st.info(
                    "**Note on OCR accuracy:** Library OCR (Tesseract) makes realistic "
                    "errors on degraded/noisy images — similar to our custom ResVGG CNN "
                    "which achieves **87.86% character-level accuracy** on EMNIST (62 classes). "
                    "Both approaches struggle with visually similar characters: `0/O`, `1/l/I`, `5/S`. "
                    "The compression stage is **always lossless** — whatever the OCR extracts "
                    "is perfectly preserved through the Adaptive Huffman encoder."
                )

else:
    st.info("⬆️ Upload an image or pick an example from the sidebar to get started.")

st.divider()
st.caption(
    "Stage 1: Tesseract OCR (library) · "
    "Stage 2: Custom Adaptive Huffman — online, no zlib, tree updates per symbol"
)
