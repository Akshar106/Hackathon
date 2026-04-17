"""
Standalone evaluation script — run after training to produce the metrics
reported in the README (grad requirement).

Reports:
  1. CharClassifier — overall val accuracy on clean EMNIST
  2. CharClassifier — per-noise accuracy (Gaussian, salt-and-pepper)
  3. DenoisingUNet  — PSNR and SSIM on SimulatedNoisyOffice test split

Usage:
    python training/evaluate.py
    python training/evaluate.py --n-samples 5000
"""

import argparse
import os
import sys
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ocr_service.model import CharClassifier, DenoisingUNet

DATA_ROOT  = os.path.join(os.path.dirname(__file__), "..", "SimulatedNoisyOffice")
DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
NOISY_DIR  = os.path.join(DATA_ROOT, "simulated_noisy_images_grayscale")
CLEAN_DIR  = os.path.join(DATA_ROOT, "clean_images_grayscale")
NUM_CLASSES = 62


# ── Noise transforms (mirrored from train_classifier) ────────────────────────

class AddGaussianNoise:
    def __init__(self, sigma: float = 0.10):
        self.sigma = sigma

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return (t + self.sigma * torch.randn_like(t)).clamp(0.0, 1.0)


class AddSaltAndPepperNoise:
    def __init__(self, density: float = 0.05):
        self.density = density

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        mask = torch.rand_like(t)
        out = t.clone()
        out[mask < self.density / 2] = 0.0
        out[mask > 1 - self.density / 2] = 1.0
        return out


# ── PSNR / SSIM helpers ───────────────────────────────────────────────────────

def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float("inf")
    return 10 * math.log10(1.0 / mse)


def ssim_score(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> float:
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_x = F.avg_pool2d(pred,   window_size, 1, window_size // 2)
    mu_y = F.avg_pool2d(target, window_size, 1, window_size // 2)
    mu_x2, mu_y2, mu_xy = mu_x**2, mu_y**2, mu_x * mu_y
    sig_x  = F.avg_pool2d(pred**2,    window_size, 1, window_size // 2) - mu_x2
    sig_y  = F.avg_pool2d(target**2,  window_size, 1, window_size // 2) - mu_y2
    sig_xy = F.avg_pool2d(pred*target, window_size, 1, window_size // 2) - mu_xy
    ssim_map = ((2*mu_xy+C1)*(2*sig_xy+C2)) / ((mu_x2+mu_y2+C1)*(sig_x+sig_y+C2))
    return ssim_map.mean().item()


# ── Section 1: CharClassifier ─────────────────────────────────────────────────

def eval_classifier(device: str, n_samples: int):
    weights = os.path.join(MODEL_DIR, "char_classifier.pth")
    if not os.path.exists(weights):
        print("[SKIP] char_classifier.pth not found — run train_classifier.py first.")
        return

    model = CharClassifier(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    val_ds = torchvision.datasets.EMNIST(
        root=DATA_DIR, split="byclass", train=False,
        download=True,
        transform=T.Compose([T.Resize((32, 32)), T.ToTensor()]),
    )

    # Overall accuracy on clean images
    loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=4)
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            preds = model(imgs.to(device)).argmax(1)
            correct += (preds == labels.to(device)).sum().item()
            total   += labels.size(0)
    clean_acc = correct / total
    print(f"\n── CharClassifier ───────────────────────────────────────")
    print(f"  Clean EMNIST val accuracy : {clean_acc:.4f}  ({correct}/{total})")

    # Per-noise accuracy
    indices = np.random.choice(len(val_ds), min(n_samples, len(val_ds)), replace=False)
    imgs_clean = torch.stack([val_ds[i][0] for i in indices])
    labels     = torch.tensor([val_ds[i][1] for i in indices])

    noise_fns = {
        "Gaussian (σ=0.10)"       : AddGaussianNoise(sigma=0.10),
        "Salt & Pepper (d=0.05)"  : AddSaltAndPepperNoise(density=0.05),
    }

    with torch.no_grad():
        for name, fn in noise_fns.items():
            noisy = torch.stack([fn(img) for img in imgs_clean]).to(device)
            preds = model(noisy).argmax(1)
            acc   = (preds == labels.to(device)).float().mean().item()
            print(f"  {name:30s}: {acc:.4f}")

    if clean_acc < 0.95:
        print("  ⚠ Accuracy below 95% target.")


# ── Section 2: DenoisingUNet ──────────────────────────────────────────────────

def eval_denoiser(device: str):
    weights = os.path.join(MODEL_DIR, "denoiser.pth")
    if not os.path.exists(weights):
        print("\n[SKIP] denoiser.pth not found — run train_denoiser.py first.")
        return

    model = DenoisingUNet(base_ch=32).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    to_tensor = T.ToTensor()
    noisy_paths = sorted(Path(NOISY_DIR).glob("*_TE.png"))   # test split only

    if not noisy_paths:
        print("\n[SKIP] No *_TE.png images found in noisy dir.")
        return

    psnr_scores, ssim_scores = [], []
    noise_type_scores: dict = {}

    with torch.no_grad():
        for noisy_path in noisy_paths:
            parts = noisy_path.stem.split("_")
            font_code, noise_code, partition = parts[0], parts[1], parts[2]
            clean_name = f"{font_code}_Clean_{partition}.png"
            clean_path = Path(CLEAN_DIR) / clean_name
            if not clean_path.exists():
                continue

            noisy_img = to_tensor(Image.open(noisy_path).convert("L")).unsqueeze(0).to(device)
            clean_img = to_tensor(Image.open(clean_path).convert("L")).unsqueeze(0).to(device)

            pred = model(noisy_img)

            p = psnr(pred, clean_img)
            s = ssim_score(pred, clean_img)
            psnr_scores.append(p)
            ssim_scores.append(s)

            noise_label = noise_code  # e.g. 'Noisec', 'Noisef' etc.
            if noise_label not in noise_type_scores:
                noise_type_scores[noise_label] = {"psnr": [], "ssim": []}
            noise_type_scores[noise_label]["psnr"].append(p)
            noise_type_scores[noise_label]["ssim"].append(s)

    noise_names = {
        "Noisec": "coffee stains",
        "Noisef": "folded sheets",
        "Noisep": "footprints",
        "Noisew": "wrinkled",
    }

    print(f"\n── DenoisingUNet (test split, {len(psnr_scores)} images) ────────────────")
    print(f"  Overall PSNR : {np.mean(psnr_scores):.2f} dB")
    print(f"  Overall SSIM : {np.mean(ssim_scores):.4f}")
    print(f"\n  Per noise type:")
    for code, vals in sorted(noise_type_scores.items()):
        label = noise_names.get(code, code)
        print(f"    {label:20s}  PSNR={np.mean(vals['psnr']):.2f} dB  SSIM={np.mean(vals['ssim']):.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=3000,
                   help="Number of EMNIST val samples for per-noise accuracy")
    p.add_argument("--device",    type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else
                              "cpu")
    print(f"Evaluation device: {device}")
    eval_classifier(device, args.n_samples)
    eval_denoiser(device)
    print()


if __name__ == "__main__":
    main()
