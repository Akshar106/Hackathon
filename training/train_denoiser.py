import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ocr_service.model import DenoisingUNet

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "SimulatedNoisyOffice")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

NOISY_DIR = os.path.join(DATA_ROOT, "simulated_noisy_images_grayscale")
CLEAN_DIR = os.path.join(DATA_ROOT, "clean_images_grayscale")

NOISE_TYPES = {"Noisec": "coffee", "Noisef": "folded", "Noisep": "footprints", "Noisew": "wrinkled"}


class NoisyOfficeDataset(Dataset):
    """
    Pairs noisy images with their clean counterparts by matching the Font
    and partition codes in the filename.

    Noisy filename pattern : FontABC_NoiseD_EE.png
    Clean filename pattern : FontABC_Clean_EE.png
    """

    def __init__(self, noisy_dir: str, clean_dir: str, patch_size: int = 256):
        self.patch_size = patch_size
        self.pairs = []

        noisy_paths = sorted(Path(noisy_dir).glob("*.png"))
        for noisy_path in noisy_paths:
            parts = noisy_path.stem.split("_")         
            font_code = parts[0]                      
            partition  = parts[2]                    
            clean_name = f"{font_code}_Clean_{partition}.png"
            clean_path = Path(clean_dir) / clean_name
            if clean_path.exists():
                self.pairs.append((noisy_path, clean_path))

        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No matching noisy/clean pairs found.\n"
                f"  noisy_dir={noisy_dir}\n  clean_dir={clean_dir}"
            )

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        noisy_path, clean_path = self.pairs[idx]
        noisy = Image.open(noisy_path).convert("L")
        clean = Image.open(clean_path).convert("L")

        noisy, clean = self._random_crop(noisy, clean)

        return self.to_tensor(noisy), self.to_tensor(clean)

    def _random_crop(self, noisy: Image.Image, clean: Image.Image):
        w, h = noisy.size
        ps = self.patch_size
        if w <= ps or h <= ps:
            noisy = noisy.resize((ps + 1, ps + 1), Image.BILINEAR)
            clean = clean.resize((ps + 1, ps + 1), Image.BILINEAR)
            w, h = ps + 1, ps + 1

        import random
        x = random.randint(0, w - ps)
        y = random.randint(0, h - ps)
        box = (x, y, x + ps, y + ps)
        return noisy.crop(box), clean.crop(box)

def ssim_loss(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """1 - SSIM, so minimising this maximises structural similarity."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_x = F.avg_pool2d(pred,   window_size, 1, window_size // 2)
    mu_y = F.avg_pool2d(target, window_size, 1, window_size // 2)
    mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y

    sig_x  = F.avg_pool2d(pred   ** 2, window_size, 1, window_size // 2) - mu_x2
    sig_y  = F.avg_pool2d(target ** 2, window_size, 1, window_size // 2) - mu_y2
    sig_xy = F.avg_pool2d(pred * target, window_size, 1, window_size // 2) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sig_xy + C2)) / \
               ((mu_x2 + mu_y2 + C1) * (sig_x + sig_y + C2))
    return 1.0 - ssim_map.mean()


def combined_loss(pred, target, alpha: float = 0.8):
    """alpha * MSE + (1-alpha) * SSIM_loss."""
    return alpha * F.mse_loss(pred, target) + (1 - alpha) * ssim_loss(pred, target)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for noisy, clean in loader:
        noisy, clean = noisy.to(device), clean.to(device)
        optimizer.zero_grad()
        pred = model(noisy)
        loss = combined_loss(pred, clean)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * noisy.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    for noisy, clean in loader:
        noisy, clean = noisy.to(device), clean.to(device)
        pred = model(noisy)
        total_loss += combined_loss(pred, clean).item() * noisy.size(0)
    return total_loss / len(loader.dataset)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch-size", type=int,   default=4)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--patch-size", type=int,   default=256)
    p.add_argument("--val-split",  type=float, default=0.15)
    p.add_argument("--device",     type=str,   default="")
    return p.parse_args()


def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else
                             "mps"  if torch.backends.mps.is_available() else
                             "cpu")
    print(f"Using device: {device}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    dataset = NoisyOfficeDataset(NOISY_DIR, CLEAN_DIR, patch_size=args.patch_size)
    print(f"Dataset: {len(dataset)} noisy/clean pairs")

    n_val = max(1, int(len(dataset) * args.val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = DenoisingUNet(base_ch=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5, verbose=True
    )

    best_val_loss = float("inf")
    save_path = os.path.join(MODEL_DIR, "denoiser.pth")

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device)
        va_loss = validate(model, val_loader, device)
        scheduler.step(va_loss)

        print(f"Epoch {epoch:03d}/{args.epochs}  train_loss={tr_loss:.5f}  val_loss={va_loss:.5f}")

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved best denoiser (val_loss={va_loss:.5f})")

    print(f"\nDenoiser training complete. Best val loss: {best_val_loss:.5f}")
    print(f"Weights saved to: {save_path}")


if __name__ == "__main__":
    main()
