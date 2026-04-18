import torch
import torch.nn as nn


class CharClassifier(nn.Module):
    """
    VGG-style CNN for 62-class EMNIST character classification.

    Architecture (32×32 input):
      Block1: Conv(1→64)  × 2, BN, ReLU, MaxPool2d → 16×16
      Block2: Conv(64→128)× 2, BN, ReLU, MaxPool2d →  8×8
      Block3: Conv(128→256)×2, BN, ReLU, MaxPool2d →  4×4
      GAP   : 256-dim vector
      Head  : 256 → 512 → num_classes

    Wider channels (64/128/256 vs old 32/64/128) give more capacity
    for the 62 visually-similar EMNIST classes while keeping the proven
    VGG-style structure that runs stably on BigRed200 with cudnn disabled.
    """

    def __init__(self, num_classes: int = 62):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        return self.classifier(x)


class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DenoisingUNet(nn.Module):
    def __init__(self, base_ch: int = 32):
        super().__init__()

        self.enc1 = _DoubleConv(1, base_ch)
        self.enc2 = _DoubleConv(base_ch, base_ch * 2)
        self.enc3 = _DoubleConv(base_ch * 2, base_ch * 4)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = _DoubleConv(base_ch * 4, base_ch * 8)

        self.up3  = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec3 = _DoubleConv(base_ch * 8, base_ch * 4)

        self.up2  = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = _DoubleConv(base_ch * 4, base_ch * 2)

        self.up1  = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = _DoubleConv(base_ch * 2, base_ch)

        self.out_conv = nn.Sequential(
            nn.Conv2d(base_ch, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _match_size(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        th, tw = target.size(2), target.size(3)
        x = x[:, :, :th, :tw]
        dh = th - x.size(2)
        dw = tw - x.size(3)
        if dh > 0 or dw > 0:
            x = torch.nn.functional.pad(x, [0, dw, 0, dh])
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self._match_size(self.up3(b),  e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._match_size(self.up2(d3), e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._match_size(self.up1(d2), e1), e1], dim=1))

        return self.out_conv(d1)
