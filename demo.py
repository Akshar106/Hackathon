"""
Hackathon Demo — 2-Stage Neural Compression Pipeline
======================================================
Streamlit UI showing the full pipeline:
  Image → OCR (CNN) → Compressed bitstream → Decompressed text recovery

Run:
    # Start both microservices first (two separate terminals):
    uvicorn ocr_service.api:app --port 8000
    uvicorn compression_service.api:app --port 8001

    # Then launch the demo:
    streamlit run demo.py
"""

import io
import os
import time
from pathlib import Path

import requests
import streamlit as st
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────

OCR_URL      = os.environ.get("OCR_URL",      "http://localhost:8000")
COMPRESS_URL = os.environ.get("COMPRESS_URL", "http://localhost:8001")
TEST_IMG_DIR = Path(__file__).parent / "test_images"

# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Neural Compression Pipeline",
    page_icon="🗜️",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
.pipeline-arrow {
    text-align: center;
    font-size: 2rem;
    color: #666;
    padding: 0.2rem 0;
}
.metric-box {
    background: #f0f2f6;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.3rem 0;
}
.success-box {
    background: #d4edda;
    border: 1px solid #c3e6cb;
    border-radius: 8px;
    padding: 1rem;
    color: #155724;
}
.error-box {
    background: #f8d7da;
    border: 1px solid #f5c6cb;
    border-radius: 8px;
    padding: 1rem;
    color: #721c24;
}
.step-header {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.bitstream-box {
    font-family: monospace;
    font-size: 0.75rem;
    background: #1e1e1e;
    color: #4ec9b0;
    border-radius: 6px;
    padding: 0.8rem;
    word-break: break-all;
    max-height: 120px;
    overflow-y: auto;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def check_service(url: str) -> bool:
    try:
        r = requests.get(f"{url}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def call_ocr(image_bytes: bytes, filename: str, denoise: bool) -> dict:
    resp = requests.post(
        f"{OCR_URL}/ocr",
        files={"file": (filename, image_bytes, "image/png")},
        params={"denoise": str(denoise).lower()},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def call_compress(text: str) -> dict:
    text_bytes = text.encode("utf-8")
    resp = requests.post(
        f"{COMPRESS_URL}/compress",
        files={"file": ("ocr_output.txt", text_bytes, "text/plain")},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def call_decompress(bitstring: str, original_length: int) -> dict:
    resp = requests.post(
        f"{COMPRESS_URL}/decompress",
        json={
            "bitstring":       bitstring,
            "original_length": original_length,
            "freqs":           {},
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def get_test_images():
    if not TEST_IMG_DIR.exists():
        return []
    return sorted([
        p for p in TEST_IMG_DIR.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    ])


# ── Header ────────────────────────────────────────────────────────────────────

st.title("🗜️ 2-Stage Neural Compression Pipeline")
st.markdown(
    "**Stage 1 → OCR** (ResVGG CNN on EMNIST, 62 classes) &nbsp;|&nbsp; "
    "**Stage 2 → Adaptive Huffman Encoding** (custom, no zlib)"
)

# ── Service status banner ─────────────────────────────────────────────────────

with st.container():
    c1, c2 = st.columns(2)
    ocr_ok  = check_service(OCR_URL)
    comp_ok = check_service(COMPRESS_URL)

    with c1:
        if ocr_ok:
            st.success(f"✅ OCR Service — {OCR_URL}")
        else:
            st.error(f"❌ OCR Service offline — {OCR_URL}")

    with c2:
        if comp_ok:
            st.success(f"✅ Compression Service — {COMPRESS_URL}")
        else:
            st.error(f"❌ Compression Service offline — {COMPRESS_URL}")

    if not (ocr_ok and comp_ok):
        st.warning(
            "Start both services before running the demo:\n"
            "```\nuvicorn ocr_service.api:app --port 8000\n"
            "uvicorn compression_service.api:app --port 8001\n```"
        )

st.divider()

# ── Sidebar settings ──────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")
    denoise_enabled = st.toggle("Enable Denoising (UNet)", value=True)
    st.caption("Applies DenoisingUNet before OCR. Helps with noisy scans.")

    st.divider()
    st.header("🖼️ Example Images")
    test_imgs = get_test_images()
    if test_imgs:
        selected_example = st.selectbox(
            "Load a test image",
            options=["— choose —"] + [p.name for p in test_imgs],
        )
    else:
        selected_example = "— choose —"
        st.caption("No test images found in test_images/")

    st.divider()
    st.header("ℹ️ Architecture")
    st.markdown("""
**OCR Model**
- 4-block ResVGG CNN
- 1→64→128→256→512 channels
- Residual skip connections
- 32×32 grayscale input
- 62 classes (0-9, A-Z, a-z)
- 5.15M parameters

**Compression**
- Adaptive Huffman (online)
- No pre-scan required
- Tree updates per symbol
- No zlib / external libs
""")

# ── Pipeline ──────────────────────────────────────────────────────────────────

# ── Step 0: Image input ───────────────────────────────────────────────────────

st.subheader("📥 Step 1 — Upload Image")

uploaded_file = st.file_uploader(
    "Upload a scanned document image (PNG, JPEG, BMP)",
    type=["png", "jpg", "jpeg", "bmp", "tiff"],
)

# Load from example if selected
if selected_example != "— choose —" and uploaded_file is None:
    example_path = TEST_IMG_DIR / selected_example
    with open(example_path, "rb") as f:
        example_bytes = f.read()
    uploaded_file = io.BytesIO(example_bytes)
    uploaded_file.name = selected_example

# Show image and run button
if uploaded_file is not None:
    image_bytes = uploaded_file.read() if hasattr(uploaded_file, "read") else open(uploaded_file, "rb").read()
    filename    = getattr(uploaded_file, "name", "image.png")

    col_img, col_btn = st.columns([2, 1])
    with col_img:
        img = Image.open(io.BytesIO(image_bytes))
        st.image(img, caption=f"{filename}  ({img.width}×{img.height}px)", use_container_width=True)

    with col_btn:
        st.markdown("&nbsp;")
        run_pipeline = st.button(
            "▶ Run Full Pipeline",
            type="primary",
            disabled=not (ocr_ok and comp_ok),
            use_container_width=True,
        )
        if not (ocr_ok and comp_ok):
            st.caption("Start both services to enable.")

    # ── Run pipeline ──────────────────────────────────────────────────────────

    if run_pipeline:
        st.divider()

        # ── Stage 1: OCR ──────────────────────────────────────────────────────

        st.subheader("🔍 Stage 1 — OCR (CNN Character Recognition)")

        with st.spinner("Running OCR..."):
            t_ocr_start = time.perf_counter()
            try:
                ocr_result  = call_ocr(image_bytes, filename, denoise_enabled)
                ocr_latency = (time.perf_counter() - t_ocr_start) * 1000
                ocr_ok_run  = True
            except Exception as exc:
                st.error(f"OCR failed: {exc}")
                ocr_ok_run = False

        if ocr_ok_run:
            extracted_text = ocr_result["text"]
            num_chars      = ocr_result["num_chars"]

            c1, c2, c3 = st.columns(3)
            c1.metric("Characters extracted", num_chars)
            c2.metric("OCR latency",           f"{ocr_result['latency_ms']:.1f} ms")
            c3.metric("Denoising",             "ON" if ocr_result["denoise_enabled"] else "OFF")

            st.markdown("**Extracted text:**")
            st.code(extracted_text if extracted_text else "(no text detected)", language=None)

            if not extracted_text:
                st.warning(
                    "No characters detected. Try a clearer image or disable denoising."
                )
                st.stop()

            # ── Stage 1 → Stage 2 arrow ───────────────────────────────────────

            st.markdown(
                '<div class="pipeline-arrow">⬇️ OCR text → Compression</div>',
                unsafe_allow_html=True,
            )

            # ── Stage 2: Compression ──────────────────────────────────────────

            st.subheader("🗜️ Stage 2 — Adaptive Huffman Compression")

            with st.spinner("Compressing..."):
                try:
                    comp_result = call_compress(extracted_text)
                    comp_ok_run = True
                except Exception as exc:
                    st.error(f"Compression failed: {exc}")
                    comp_ok_run = False

            if comp_ok_run:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Original bytes",     comp_result["original_bytes"])
                c2.metric("Compressed bytes",   comp_result["compressed_bytes"])
                c3.metric("Compression ratio",  f"{comp_result['compression_ratio']:.2f}×")
                c4.metric("Encoding efficiency",f"{comp_result['encoding_efficiency']*100:.1f}%")

                c5, c6 = st.columns(2)
                c5.metric("Entropy (bits/sym)", f"{comp_result['entropy_bits_per_symbol']:.3f}")
                c6.metric("Avg bits/symbol",    f"{comp_result['avg_bits_per_symbol']:.3f}")

                # Show bitstream preview
                bitstring = comp_result["bitstring"]
                st.markdown("**Compressed bitstream (preview):**")
                preview = bitstring[:300] + ("..." if len(bitstring) > 300 else "")
                st.markdown(
                    f'<div class="bitstream-box">{preview}</div>',
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"Total bitstream length: {len(bitstring):,} bits "
                    f"({comp_result['compressed_bytes']:,} bytes packed)"
                )

                # ── Stage 2 → Recovery arrow ──────────────────────────────────

                st.markdown(
                    '<div class="pipeline-arrow">⬇️ Bitstream → Decompression</div>',
                    unsafe_allow_html=True,
                )

                # ── Stage 3: Decompression ────────────────────────────────────

                st.subheader("♻️ Stage 3 — Decompression & Recovery Verification")

                with st.spinner("Decompressing..."):
                    try:
                        decomp_result = call_decompress(
                            bitstring,
                            comp_result["original_length"],
                        )
                        decomp_ok_run = True
                    except Exception as exc:
                        st.error(f"Decompression failed: {exc}")
                        decomp_ok_run = False

                if decomp_ok_run:
                    recovered_text = decomp_result["text"]
                    match = recovered_text == extracted_text

                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Decompressed text:**")
                        st.code(recovered_text, language=None)

                    with c2:
                        if match:
                            st.markdown(
                                '<div class="success-box">'
                                '<b>✅ Perfect recovery!</b><br>'
                                'Decompressed text is byte-for-byte identical to OCR output.'
                                '</div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                '<div class="error-box">'
                                '<b>❌ Recovery mismatch</b><br>'
                                'Decompressed text differs from original OCR output.'
                                '</div>',
                                unsafe_allow_html=True,
                            )

                        st.metric(
                            "Original length",   comp_result["original_length"]
                        )
                        st.metric(
                            "Recovered length",  decomp_result["recovered_length"]
                        )
                        st.metric(
                            "Decompress latency",
                            f"{decomp_result['latency_ms']:.1f} ms",
                        )

                    # ── Full pipeline summary ─────────────────────────────────

                    st.divider()
                    st.subheader("📊 Full Pipeline Summary")

                    total_latency = (
                        ocr_result["latency_ms"]
                        + comp_result["latency_ms"]
                        + decomp_result["latency_ms"]
                    )

                    s1, s2, s3, s4 = st.columns(4)
                    s1.metric("OCR latency",         f"{ocr_result['latency_ms']:.1f} ms")
                    s2.metric("Compress latency",    f"{comp_result['latency_ms']:.1f} ms")
                    s3.metric("Decompress latency",  f"{decomp_result['latency_ms']:.1f} ms")
                    s4.metric("Total latency",       f"{total_latency:.1f} ms")

                    st.markdown("### Pipeline Flow")
                    st.markdown(f"""
```
┌─────────────────────────────────────────────────────────────────┐
│  INPUT IMAGE ({img.width}×{img.height} px)                     │
│       ↓  OCR — ResVGG CNN  ({ocr_result['latency_ms']:.0f} ms) │
│  EXTRACTED TEXT  [{num_chars} chars]: "{extracted_text[:40]}{'...' if len(extracted_text)>40 else ''}"
│       ↓  Adaptive Huffman Compress ({comp_result['latency_ms']:.0f} ms)
│  BITSTREAM  [{len(bitstring):,} bits → {comp_result['compressed_bytes']} bytes | ratio {comp_result['compression_ratio']:.2f}×]
│       ↓  Adaptive Huffman Decompress ({decomp_result['latency_ms']:.0f} ms)
│  RECOVERED TEXT  [{decomp_result['recovered_length']} chars]  {"✅ PERFECT MATCH" if match else "❌ MISMATCH"}
└─────────────────────────────────────────────────────────────────┘
```
""")

else:
    st.info("⬆️ Upload an image or choose an example from the sidebar to get started.")

# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Stage 1: 4-block ResVGG CNN (5.15M params) trained on EMNIST byclass (62 classes) · "
    "Stage 2: Custom Adaptive Huffman — online, no zlib, tree updates per symbol"
)
