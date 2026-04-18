"""
Generate synthetic scanned document test images.

Creates realistic-looking grayscale document images with:
  - Printed text (letters, digits, mixed)
  - Optional office noise (Gaussian, salt & pepper, coffee stain simulation)

Output saved to: test_images/

Usage:
    python generate_test_image.py
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_images")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_font(size=28):
    """Try to load a system font, fall back to default."""
    candidates = [
        "/System/Library/Fonts/Courier.ttc",
        "/System/Library/Fonts/Monaco.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def make_document(text_lines, width=640, height=480, font_size=24):
    """Create a clean white document image with printed text."""
    img = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(img)
    font = get_font(font_size)

    margin_x, margin_y = 40, 40
    line_spacing = font_size + 10
    y = margin_y

    for line in text_lines:
        draw.text((margin_x, y), line, fill=0, font=font)
        y += line_spacing
        if y > height - margin_y:
            break

    return img


def add_gaussian_noise(img, sigma=15):
    arr = np.array(img, dtype=np.float32)
    arr += np.random.normal(0, sigma, arr.shape)
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def add_salt_pepper(img, density=0.04):
    arr = np.array(img).copy()
    n = int(arr.size * density)
    # Salt (white dots)
    coords = [np.random.randint(0, d, n) for d in arr.shape]
    arr[coords[0], coords[1]] = 255
    # Pepper (black dots)
    coords = [np.random.randint(0, d, n) for d in arr.shape]
    arr[coords[0], coords[1]] = 0
    return Image.fromarray(arr)


def add_coffee_stain(img):
    """Simulate a faint circular coffee stain."""
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape
    cx, cy = random.randint(w//4, 3*w//4), random.randint(h//4, 3*h//4)
    radius = random.randint(40, 90)
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    stain = np.clip(1 - dist / radius, 0, 1) * random.uniform(20, 50)
    arr = np.clip(arr - stain, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


# ── Document content ──────────────────────────────────────────────────────────

DOCUMENTS = {
    "clean": [
        "Invoice #A1042",
        "Date: 2024-11-15",
        "Bill To: John Smith",
        "Address: 123 Main Street",
        "Item        Qty   Price",
        "Widget A     2    $14.99",
        "Widget B     5    $8.50",
        "Subtotal:         $72.48",
        "Tax (8%):          $5.80",
        "Total:            $78.28",
    ],
    "mixed_chars": [
        "ABCDEFGHIJKLM",
        "NOPQRSTUVWXYZ",
        "abcdefghijklm",
        "nopqrstuvwxyz",
        "0123456789",
        "Hello World 2024",
        "The Quick Brown Fox",
        "Jumps Over The Dog",
        "Score: 97 / 100",
        "ID: X7K9-P2Q4",
    ],
    "letter": [
        "Dear Dr. Johnson,",
        "",
        "I am writing to confirm",
        "our meeting scheduled for",
        "Monday, December 2nd at",
        "10:30 AM in Room B204.",
        "",
        "Please bring the Q3 report",
        "and budget projections.",
        "",
        "Best regards,",
        "A. Kumar",
    ],
}


def generate_all():
    saved = []

    for doc_name, lines in DOCUMENTS.items():
        base_img = make_document(lines)

        # 1. Clean version
        path = os.path.join(OUTPUT_DIR, f"{doc_name}_clean.png")
        base_img.save(path)
        saved.append(path)

        # 2. Gaussian noise
        path = os.path.join(OUTPUT_DIR, f"{doc_name}_gaussian.png")
        add_gaussian_noise(base_img, sigma=18).save(path)
        saved.append(path)

        # 3. Salt & pepper
        path = os.path.join(OUTPUT_DIR, f"{doc_name}_salt_pepper.png")
        add_salt_pepper(base_img, density=0.04).save(path)
        saved.append(path)

        # 4. Coffee stain
        path = os.path.join(OUTPUT_DIR, f"{doc_name}_coffee.png")
        add_coffee_stain(base_img).save(path)
        saved.append(path)

        # 5. Combined (Gaussian + coffee stain) — hardest
        path = os.path.join(OUTPUT_DIR, f"{doc_name}_combined.png")
        noisy = add_gaussian_noise(add_coffee_stain(base_img), sigma=12)
        noisy.save(path)
        saved.append(path)

    print(f"Generated {len(saved)} test images in: {OUTPUT_DIR}/")
    for p in saved:
        print(f"  {os.path.basename(p)}")


if __name__ == "__main__":
    generate_all()
