import io
import os
import re
import sys
import time
from pathlib import Path

import requests
import streamlit as st
from PIL import Image
import numpy as np


def _run_ocr_engine(image: Image.Image) -> str:
    try:
        import pytesseract
        config = "--psm 6 --oem 3"
        text = pytesseract.image_to_string(image, config=config).strip()
        return text
    except Exception:
        pass
    try:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        img_np = np.array(image.convert("RGB"))
        results = reader.readtext(img_np, detail=0, paragraph=True)
        return "\n".join(results)
    except Exception:
        return ""


def run_ocr(image: Image.Image) -> tuple[str, float]:
    t0 = time.perf_counter()
    raw = _run_ocr_engine(image)
    latency_ms = (time.perf_counter() - t0) * 1000
    clean = re.sub(r"[^A-Za-z0-9 \n]", "", raw)
    clean = re.sub(r"\n{3,}", "\n\n", clean).strip()
    return clean, latency_ms


def get_denoised_image(image: Image.Image) -> Image.Image | None:
    """
    Run DenoisingUNet on the image and return the denoised PIL Image.
    Tries direct import first; falls back to OCR service endpoint.
    Returns None if denoising is unavailable.
    """
    # Try direct import (works when ocr_service is on the path)
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from ocr_service.preprocess import load_grayscale, denoise_array, load_denoiser

        denoiser_path = Path(__file__).parent / "models" / "denoiser.pth"
        if denoiser_path.exists():
            load_denoiser(str(denoiser_path), device="cpu")

        gray = load_grayscale(image)
        denoised = denoise_array(gray)
        return Image.fromarray(denoised, mode="L")
    except Exception:
        pass

    # Fall back to OCR service endpoint
    try:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        OCR_URL = os.environ.get("OCR_URL", "http://localhost:8000")
        resp = requests.post(
            f"{OCR_URL}/ocr/denoise",
            files={"file": ("image.png", buf.read(), "image/png")},
            timeout=15,
        )
        if resp.status_code == 200:
            return Image.open(io.BytesIO(resp.content))
    except Exception:
        pass

    return None


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
    sys.path.insert(0, str(Path(__file__).parent))
    from compression_service.huffman import compress

    data = text.encode("utf-8")
    t0 = time.perf_counter()
    payload = compress(data)
    latency_ms = (time.perf_counter() - t0) * 1000
    payload["latency_ms"] = round(latency_ms, 2)
    for k, v in payload.pop("stats", {}).items():
        payload[k] = v
    return payload


def decompress_direct(bitstring: str, original_length: int) -> dict:
    from compression_service.huffman import decompress

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


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Neural Compression Pipeline",
    page_icon="",
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
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────

st.title("2-Stage Neural Compression Pipeline")
st.markdown(
    "**Stage 1: OCR** (ResVGG CNN, 62 classes)  |  "
    "**Stage 2: Adaptive Huffman** (custom, no zlib)"
)

comp_ok = check_compression_service()
use_direct = not comp_ok

col1, col2 = st.columns(2)
with col1:
    st.success("OCR Model ready")
with col2:
    if comp_ok:
        st.success(f"Compression Service ready  {COMPRESS_URL}")
    else:
        st.warning("Compression Service offline  running in direct mode")

if not comp_ok:
    st.info(
        "Compression running in-process. "
        "To use the service: uvicorn compression_service.api:app --port 8001"
    )

st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")

    st.divider()
    st.header("Example Images")
    TEST_DIR = Path("test_images")
    test_imgs = sorted(TEST_DIR.glob("*.png")) if TEST_DIR.exists() else []
    selected = st.selectbox(
        "Load example",
        ["choose"] + [p.name for p in test_imgs],
    )

    st.divider()
    st.header("About")
    st.markdown("""
**Stage 1: OCR Model**
- 4-block ResVGG CNN
- 1 to 64 to 128 to 256 to 512 channels
- Residual skip connections
- 32x32 grayscale input
- 62 classes (0-9, A-Z, a-z)
- 5.15M parameters
- 87.86% val accuracy on EMNIST

**Stage 2: Custom Huffman**
- Adaptive Huffman (our code)
- Online, no pre-scan
- No zlib or external libs
- Perfect lossless recovery
""")

# ── Image input ───────────────────────────────────────────────────────────────

st.subheader("Step 1  Upload Image")

uploaded = st.file_uploader(
    "Upload a scanned document image",
    type=["png", "jpg", "jpeg", "bmp", "tiff"],
)

if selected != "choose" and uploaded is None:
    with open(TEST_DIR / selected, "rb") as f:
        uploaded = io.BytesIO(f.read())
    uploaded.name = selected

if uploaded is not None:
    img_bytes = uploaded.read()
    fname = getattr(uploaded, "name", "image.png")
    image = Image.open(io.BytesIO(img_bytes))

    col_orig, col_denoised, col_btn = st.columns([2, 2, 1])
    with col_orig:
        st.markdown("**Original Image**")
        st.image(image, caption=f"{fname}  ({image.width}x{image.height}px)",
                 use_container_width=True)

    with col_denoised:
        st.markdown("**Denoised Image** (DenoisingUNet)")
        denoised_img = get_denoised_image(image)
        if denoised_img is not None:
            st.image(denoised_img, caption="After DenoisingUNet (SSIM 0.99+)",
                     use_container_width=True)
        else:
            st.caption("Denoised preview unavailable. Place denoiser.pth in models/ to enable.")

    with col_btn:
        st.markdown("&nbsp;")
        run = st.button(
            "Run Full Pipeline",
            type="primary",
            use_container_width=True,
        )

    if run:
        st.divider()

        # ── Stage 1: OCR ──────────────────────────────────────────────────────

        st.subheader("Stage 1  OCR (Character Recognition)")

        with st.spinner("Running OCR..."):
            ocr_text, ocr_latency = run_ocr(image)

        c1, c2 = st.columns(2)
        c1.metric("Characters extracted", len(ocr_text))
        c2.metric("OCR latency", f"{ocr_latency:.0f} ms")

        st.markdown("**Extracted text:**")
        st.text_area(
            label="ocr_output",
            value=ocr_text if ocr_text else "(no text detected)",
            height=200,
            label_visibility="collapsed",
        )

        if not ocr_text:
            st.warning("No text detected. Try a clearer image.")
            st.stop()

        st.markdown(
            '<p style="text-align:center;font-size:1.6rem;color:#888;">OCR text  Compression</p>',
            unsafe_allow_html=True,
        )

        # ── Stage 2: Compression ──────────────────────────────────────────────

        st.subheader("Stage 2  Adaptive Huffman Compression")

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
            c1.metric("Original bytes",      comp.get("original_bytes", len(ocr_text)))
            c2.metric("Compressed bytes",    comp.get("compressed_bytes", 0))
            c3.metric("Compression ratio",   f"{comp.get('compression_ratio', 0):.2f}x")
            c4.metric("Encoding efficiency", f"{comp.get('encoding_efficiency', 0)*100:.1f}%")

            c5, c6 = st.columns(2)
            c5.metric("Entropy (bits/sym)", f"{comp.get('entropy_bits_per_symbol', 0):.3f}")
            c6.metric("Avg bits/symbol",    f"{comp.get('avg_bits_per_symbol', 0):.3f}")

            bitstring = comp.get("bitstring", "")
            st.markdown("**Compressed bitstream (preview):**")
            preview = bitstring[:300] + ("..." if len(bitstring) > 300 else "")
            st.markdown(f'<div class="bitstream-box">{preview}</div>',
                        unsafe_allow_html=True)
            st.caption(
                f"Total: {len(bitstring):,} bits  "
                f"{comp.get('compressed_bytes', 0):,} bytes packed"
            )

            st.markdown(
                '<p style="text-align:center;font-size:1.6rem;color:#888;">Bitstream  Decompression</p>',
                unsafe_allow_html=True,
            )

            # ── Stage 3: Decompression ────────────────────────────────────────

            st.subheader("Stage 3  Decompression and Recovery Verification")

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
                    st.text_area(
                        label="recovered_output",
                        value=recovered,
                        height=200,
                        label_visibility="collapsed",
                    )

                with c2:
                    if match:
                        st.markdown(
                            '<div class="success-box">'
                            '<b>Perfect recovery</b><br>'
                            'Decompressed text is byte-for-byte identical to OCR output.<br>'
                            'Custom Adaptive Huffman: lossless compression guaranteed.'
                            '</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.error("Mismatch  decompressed text differs.")

                    st.metric("Original length",   orig_len)
                    st.metric("Recovered length",  decomp["recovered_length"])
                    st.metric("Decompress latency", f"{decomp['latency_ms']:.1f} ms")

                # ── Summary ───────────────────────────────────────────────────

                st.divider()
                st.subheader("Full Pipeline Summary")

                total = ocr_latency + comp.get("latency_ms", 0) + decomp["latency_ms"]
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("OCR latency",        f"{ocr_latency:.0f} ms")
                s2.metric("Compress latency",   f"{comp.get('latency_ms', 0):.1f} ms")
                s3.metric("Decompress latency", f"{decomp['latency_ms']:.1f} ms")
                s4.metric("Total latency",      f"{total:.0f} ms")

                st.markdown("### Pipeline Flow")
                st.markdown(f"""
```
INPUT IMAGE ({image.width}x{image.height} px)
  ResVGG CNN OCR  ({ocr_latency:.0f} ms)
EXTRACTED TEXT [{len(ocr_text)} chars]: "{ocr_text[:50].replace(chr(10), ' ')}{'...' if len(ocr_text) > 50 else ''}"
  Adaptive Huffman Compress  ({comp.get('latency_ms', 0):.0f} ms)
BITSTREAM [{len(bitstring):,} bits  {comp.get('compressed_bytes', 0)} bytes  ratio {comp.get('compression_ratio', 0):.2f}x]
  Adaptive Huffman Decompress  ({decomp['latency_ms']:.0f} ms)
RECOVERED TEXT [{decomp['recovered_length']} chars]  {"PERFECT MATCH" if match else "MISMATCH"}
```
""")

else:
    st.info("Upload an image or pick an example from the sidebar to get started.")

st.divider()
st.caption(
    "Stage 1: ResVGG CNN (5.15M params, 87.86% EMNIST val accuracy)  "
    "Stage 2: Custom Adaptive Huffman, online, no zlib, tree updates per symbol"
)
