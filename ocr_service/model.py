import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Character Classification CNN
# Trained on EMNIST (62 classes: 0-9, A-Z, a-z)
# Input: (B, 1, 32, 32) grayscale character patch
# Output: (B, 62) class logits
# ---------------------------------------------------------------------------
class CharClassifier(nn.Module):
    def __init__(self, num_classes: int = 62):
        super().__init__()

        # Block 1: 1 -> 32 channels, 32x32 -> 16x16
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )

        # Block 2: 32 -> 64 channels, 16x16 -> 8x8
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )

        # Block 3: 64 -> 128 channels, 8x8 -> 4x4
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )

        # Global average pooling collapses 4x4 spatial dims to 1x1,
        # making the classifier input-size agnostic beyond 32x32.
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Denoising U-Net
# Trained on SimulatedNoisyOffice (noisy -> clean grayscale image pairs)
# Input:  (B, 1, H, W) noisy grayscale document image
# Output: (B, 1, H, W) cleaned grayscale image (same spatial size)
#
# Skip connections preserve fine text detail that would otherwise be lost
# during downsampling — critical for readable character edges.
# ---------------------------------------------------------------------------
class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DenoisingUNet(nn.Module):
    def __init__(self, base_ch: int = 32):
        super().__init__()

        # Encoder
        self.enc1 = _DoubleConv(1, base_ch)
        self.enc2 = _DoubleConv(base_ch, base_ch * 2)
        self.enc3 = _DoubleConv(base_ch * 2, base_ch * 4)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = _DoubleConv(base_ch * 4, base_ch * 8)

        # Decoder — each up-block concatenates with the matching encoder skip
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec3 = _DoubleConv(base_ch * 8, base_ch * 4)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = _DoubleConv(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = _DoubleConv(base_ch * 2, base_ch)

        # Sigmoid squashes output to [0, 1] pixel range
        self.out_conv = nn.Sequential(
            nn.Conv2d(base_ch, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)
