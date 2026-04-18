"""
Preprocessing pipeline for scanned document images.

Steps:
  1. Load image and convert to grayscale
  2. Optionally run DenoisingUNet to remove office noise
  3. Binarize with Otsu thresholding
  4. Detect text lines via horizontal projection profile
  5. Segment each line into individual character bounding boxes
  6. Resize each character patch to 32x32 for CharClassifier
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ocr_service.model import DenoisingUNet


def load_grayscale(image_source) -> np.ndarray:
    """Load image from path or PIL Image into a uint8 grayscale numpy array."""
    if isinstance(image_source, (str, Path)):
        img = Image.open(image_source).convert("L")
    elif isinstance(image_source, Image.Image):
        img = image_source.convert("L")
    else:
        raise TypeError(f"Unsupported image type: {type(image_source)}")
    return np.array(img, dtype=np.uint8)


def otsu_threshold(gray: np.ndarray) -> np.ndarray:
    """Return binary image (255=foreground/text, 0=background) using Otsu's method."""
    # Compute histogram
    hist, bins = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    total = gray.size

    sum_total = np.dot(np.arange(256), hist)
    sum_bg, weight_bg = 0.0, 0.0
    best_var, best_thresh = 0.0, 0

    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var_between > best_var:
            best_var = var_between
            best_thresh = t

    # Invert: dark text on white background → text pixels become 255
    binary = np.where(gray <= best_thresh, 255, 0).astype(np.uint8)
    return binary


def find_line_rows(binary: np.ndarray, min_height: int = 12) -> List[Tuple[int, int]]:
    """
    Use horizontal projection profile to find text line row ranges.
    Returns list of (row_start, row_end) tuples.
    """
    projection = binary.sum(axis=1)  # sum of foreground pixels per row
    in_line = False
    lines = []
    start = 0

    for i, val in enumerate(projection):
        if not in_line and val > 0:
            in_line = True
            start = i
        elif in_line and val == 0:
            in_line = False
            if (i - start) >= min_height:
                lines.append((start, i))

    if in_line and (len(projection) - start) >= min_height:
        lines.append((start, len(projection)))

    return lines


def find_char_cols(
    line_binary: np.ndarray, min_width: int = 8
) -> List[Tuple[int, int]]:
    """
    Use vertical projection profile within a line to find character column ranges.
    Returns list of (col_start, col_end) tuples.
    """
    projection = line_binary.sum(axis=0)
    in_char = False
    chars = []
    start = 0

    for j, val in enumerate(projection):
        if not in_char and val > 0:
            in_char = True
            start = j
        elif in_char and val == 0:
            in_char = False
            if (j - start) >= min_width:
                chars.append((start, j))

    if in_char and (line_binary.shape[1] - start) >= min_width:
        chars.append((start, line_binary.shape[1]))

    return chars


def segment_characters(binary: np.ndarray) -> List[np.ndarray]:
    """
    Segment binary image into individual character patches.
    Returns list of uint8 grayscale patches (foreground=255).
    """
    lines = find_line_rows(binary)
    patches = []

    for r0, r1 in lines:
        line_strip = binary[r0:r1, :]
        chars = find_char_cols(line_strip)
        for c0, c1 in chars:
            patch = line_strip[:, c0:c1]
            patches.append(patch)

    return patches


def resize_patch(patch: np.ndarray, size: int = 32) -> np.ndarray:
    """Resize a character patch to size×size preserving aspect ratio with padding."""
    img = Image.fromarray(patch, mode="L")
    img.thumbnail((size, size), Image.BILINEAR)

    # Pad to exact size×size
    canvas = Image.new("L", (size, size), 0)
    x_off = (size - img.width) // 2
    y_off = (size - img.height) // 2
    canvas.paste(img, (x_off, y_off))
    return np.array(canvas, dtype=np.uint8)


# ── Denoiser integration ──────────────────────────────────────────────────────

_denoiser: Optional[DenoisingUNet] = None
_denoiser_device: str = "cpu"


def load_denoiser(weights_path: str, device: str = "cpu") -> None:
    """Load DenoisingUNet weights into module-level singleton."""
    global _denoiser, _denoiser_device
    model = DenoisingUNet(base_ch=32)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    _denoiser = model.to(device)
    _denoiser_device = device


def denoise_array(gray: np.ndarray) -> np.ndarray:
    """
    Run DenoisingUNet on a uint8 grayscale array.
    Returns denoised uint8 grayscale array (same shape).
    """
    if _denoiser is None:
        return gray  # passthrough if denoiser not loaded

    to_tensor = T.ToTensor()
    img_pil = Image.fromarray(gray, mode="L")
    x = to_tensor(img_pil).unsqueeze(0).to(_denoiser_device)  # (1,1,H,W)

    with torch.no_grad():
        out = _denoiser(x)  # (1,1,H,W) in [0,1]

    out_np = (out.squeeze().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return out_np


# ── Full preprocessing pipeline ───────────────────────────────────────────────

def preprocess_image(
    image_source,
    denoise: bool = True,
    patch_size: int = 32,
) -> List[np.ndarray]:
    """
    End-to-end preprocessing: load → denoise → binarize → segment → resize.

    Returns a list of (patch_size × patch_size) uint8 character patches.
    """
    gray = load_grayscale(image_source)

    if denoise:
        gray = denoise_array(gray)

    # Median filter removes isolated salt-and-pepper noise pixels before
    # thresholding — prevents the projection profile from seeing noise as text
    gray_pil = Image.fromarray(gray, mode="L")
    gray_pil = gray_pil.filter(ImageFilter.MedianFilter(size=3))
    gray = np.array(gray_pil, dtype=np.uint8)

    binary = otsu_threshold(gray)
    patches = segment_characters(binary)
    resized = [resize_patch(p, size=patch_size) for p in patches]
    return resized
