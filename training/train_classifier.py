import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ocr_service.model import CharClassifier

NUM_CLASSES = 62
DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

class AddGaussianNoise:
    def __init__(self, lo: float = 0.05, hi: float = 0.15):
        self.lo, self.hi = lo, hi

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        sigma = np.random.uniform(self.lo, self.hi)
        return (tensor + sigma * torch.randn_like(tensor)).clamp(0.0, 1.0)


class AddSaltAndPepperNoise:
    def __init__(self, lo: float = 0.02, hi: float = 0.08):
        self.lo, self.hi = lo, hi

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        density = np.random.uniform(self.lo, self.hi)
        mask = torch.rand_like(tensor)
        noisy = tensor.clone()
        noisy[mask < density / 2] = 0.0
        noisy[mask > 1 - density / 2] = 1.0
        return noisy


class RandomNoise:
    def __init__(self, p: float = 0.3):
        self.p = p
        self.gaussian = AddGaussianNoise()
        self.sp = AddSaltAndPepperNoise()

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if np.random.rand() > self.p:
            return tensor
        return self.gaussian(tensor) if np.random.rand() < 0.5 else self.sp(tensor)
    
def mixup_batch(images: torch.Tensor, labels: torch.Tensor, alpha: float = 0.2):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1 - lam) * images[idx]
    return mixed, labels, labels[idx], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def build_loaders(batch_size: int):
    train_tf = T.Compose([
        T.Resize((32, 32)),
        T.RandomRotation(degrees=15),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
        T.ToTensor(),
        RandomNoise(p=0.3),
    ])
    val_tf = T.Compose([
        T.Resize((32, 32)),
        T.ToTensor(),
    ])

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
        num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False,
    )
    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        mixed, y_a, y_b, lam = mixup_batch(images, labels)
        optimizer.zero_grad()
        logits = model(mixed)
        loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == y_a).sum().item()
        total += images.size(0)

        if i % 200 == 0:
            print(f"  batch {i:04d}/{len(loader)}  "
                  f"loss={loss.item():.4f}  lr={optimizer.param_groups[0]['lr']:.2e}")

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


@torch.no_grad()
def evaluate_per_noise(model, val_ds, device, n_samples: int = 2000):
    model.eval()
    gaussian = AddGaussianNoise()
    sp = AddSaltAndPepperNoise()

    n_samples = min(n_samples, len(val_ds))
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

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=60)
    p.add_argument("--batch-size", type=int,   default=256)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--device",     type=str,   default="")
    return p.parse_args()


def main():
    torch.backends.cudnn.enabled = False

    args = parse_args()

    device = args.device or (
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device : {device}")

    os.makedirs(DATA_DIR,  exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading EMNIST (byclass) ...")
    train_loader, val_loader = build_loaders(args.batch_size)
    print(f"  Train: {len(train_loader.dataset):,}  Val: {len(val_loader.dataset):,}")

    model = CharClassifier(num_classes=NUM_CLASSES).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    # Smoke test
    with torch.no_grad():
        out = model(torch.randn(2, 1, 32, 32, device=device))
        assert out.shape == (2, NUM_CLASSES)
    print("  Smoke test: PASSED")

    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # AdamW with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # OneCycleLR — warm-up for first 30% of steps, cosine decay after
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps,
        pct_start=0.3, anneal_strategy="cos",
        div_factor=25.0, final_div_factor=1e4,
    )

    best_val_acc = 0.0
    save_path = os.path.join(MODEL_DIR, "char_classifier.pth")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch:02d}/{args.epochs}")
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)

        print(f"  train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
              f"val_loss={va_loss:.4f}  val_acc={va_acc:.4f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved best model (val_acc={va_acc:.4f})")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")

    print("\nPer-noise accuracy:")
    model.load_state_dict(torch.load(save_path, map_location=device))
    noise_acc = evaluate_per_noise(model, val_loader.dataset, device)
    for name, acc in noise_acc.items():
        print(f"  {name:20s}: {acc:.4f}")

    if best_val_acc >= 0.95:
        print("\n✓ TARGET REACHED: val_acc > 95%")
    else:
        print(f"\nval_acc = {best_val_acc:.4f}  (target: 0.95)")


if __name__ == "__main__":
    main()
