"""
Train the CharClassifier CNN on EMNIST (by_class split, 62 classes).

Noise augmentation strategy (grad requirement):
  - Gaussian noise  : additive zero-mean Gaussian, sigma sampled from [0.05, 0.15]
  - Salt-and-pepper : random pixels set to 0 or 1, density sampled from [0.02, 0.08]

Both are applied randomly during training so the model sees clean AND noisy
samples, pushing it toward the >95% character-level accuracy target.

Usage:
    python training/train_classifier.py
    python training/train_classifier.py --epochs 30 --batch-size 128 --lr 1e-3
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

# Allow importing from sibling package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ocr_service.model import CharClassifier

# ── EMNIST by_class label order ──────────────────────────────────────────────
# 0-9  → digits
# 10-35 → A-Z  (uppercase)
# 36-61 → a-z  (lowercase)
NUM_CLASSES = 62
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


# ── Custom noise transforms ───────────────────────────────────────────────────

class AddGaussianNoise:
    """Additive Gaussian noise with sigma drawn uniformly from [lo, hi]."""
    def __init__(self, lo: float = 0.05, hi: float = 0.15):
        self.lo, self.hi = lo, hi

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        sigma = np.random.uniform(self.lo, self.hi)
        return (tensor + sigma * torch.randn_like(tensor)).clamp(0.0, 1.0)


class AddSaltAndPepperNoise:
    """Randomly set pixels to 0 (pepper) or 1 (salt), density in [lo, hi]."""
    def __init__(self, lo: float = 0.02, hi: float = 0.08):
        self.lo, self.hi = lo, hi

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        density = np.random.uniform(self.lo, self.hi)
        mask = torch.rand_like(tensor)
        noisy = tensor.clone()
        noisy[mask < density / 2] = 0.0        # pepper
        noisy[mask > 1 - density / 2] = 1.0    # salt
        return noisy


class RandomNoise:
    """Apply one of Gaussian or salt-and-pepper noise with probability p."""
    def __init__(self, p: float = 0.5):
        self.p = p
        self.gaussian = AddGaussianNoise()
        self.sp = AddSaltAndPepperNoise()

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if np.random.rand() > self.p:
            return tensor
        return self.gaussian(tensor) if np.random.rand() < 0.5 else self.sp(tensor)


# ── Data loaders ──────────────────────────────────────────────────────────────

def build_loaders(batch_size: int):
    # EMNIST images are 28x28; resize to 32x32 to give conv blocks
    # enough spatial room before three MaxPool2d(2) stages.
    base = [T.Resize((32, 32)), T.ToTensor()]

    train_tf = T.Compose(base + [RandomNoise(p=0.5)])
    val_tf   = T.Compose(base)

    train_ds = torchvision.datasets.EMNIST(
        root=DATA_DIR, split="byclass", train=True,
        download=True, transform=train_tf,
    )
    val_ds = torchvision.datasets.EMNIST(
        root=DATA_DIR, split="byclass", train=False,
        download=True, transform=val_tf,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    return train_loader, val_loader


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


# ── Noise-specific accuracy (grad requirement) ────────────────────────────────

@torch.no_grad()
def evaluate_per_noise(model, val_ds, device, n_samples: int = 2000):
    """
    Report accuracy separately for Gaussian and salt-and-pepper noise.
    Samples a random subset of the validation set and applies each noise type.
    """
    model.eval()
    gaussian = AddGaussianNoise()
    sp = AddSaltAndPepperNoise()

    indices = np.random.choice(len(val_ds), n_samples, replace=False)
    images = torch.stack([val_ds[i][0] for i in indices])
    labels = torch.tensor([val_ds[i][1] for i in indices])

    results = {}
    for name, noise_fn in [("gaussian", gaussian), ("salt_and_pepper", sp)]:
        noisy = torch.stack([noise_fn(img) for img in images]).to(device)
        logits = model(noisy)
        acc = (logits.argmax(1) == labels.to(device)).float().mean().item()
        results[name] = acc

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=20)
    p.add_argument("--batch-size", type=int,   default=128)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--device",     type=str,   default="")
    return p.parse_args()


def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else
                             "mps"  if torch.backends.mps.is_available() else
                             "cpu")
    print(f"Using device: {device}")

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading EMNIST (byclass) ...")
    train_loader, val_loader = build_loaders(args.batch_size)

    model = CharClassifier(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Halve LR if val loss doesn't improve for 3 consecutive epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5, verbose=True
    )

    best_val_acc = 0.0
    save_path = os.path.join(MODEL_DIR, "char_classifier.pth")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(va_loss)

        print(
            f"Epoch {epoch:02d}/{args.epochs}  "
            f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
            f"val_loss={va_loss:.4f}  val_acc={va_acc:.4f}"
        )

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved best model (val_acc={va_acc:.4f})")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")

    # Per-noise accuracy report (grad requirement)
    print("\nEvaluating per noise type on validation set ...")
    model.load_state_dict(torch.load(save_path, map_location=device))
    noise_acc = evaluate_per_noise(model, val_loader.dataset, device)
    for noise_type, acc in noise_acc.items():
        print(f"  {noise_type:20s}: {acc:.4f}")

    if best_val_acc < 0.95:
        print("\nWARNING: val accuracy below 95% target. Consider more epochs or tuning LR.")


if __name__ == "__main__":
    main()
