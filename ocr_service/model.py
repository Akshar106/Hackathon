import torch
import torch.nn as nn


class _ResVGGBlock(nn.Module):
    """
    VGG-style double-conv block with a residual skip connection.

    Conv → BN → ReLU → Conv → BN → (+skip) → ReLU → MaxPool → Dropout

    The 1×1 skip projection aligns channels when in_ch != out_ch.
    Residual connections stabilise gradients across 4 blocks and allow
    the model to learn incremental refinements rather than full mappings,
    which is important for the visually-similar 62-class EMNIST task.
    """

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.15):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        # 1×1 projection to match channels; identity when in_ch == out_ch
        self.skip = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        ) if in_ch != out_ch else nn.Identity()

        self.act  = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.conv1(x))
        out = self.act(out + self.skip(x))   # residual add before activation
        return self.drop(self.pool(out))


class CharClassifier(nn.Module):
    """
    4-block ResVGG for 62-class EMNIST character classification.

    Architecture (32×32 grayscale input):
      Block1 : ResVGGBlock( 1 →  64)  32×32 → 16×16
      Block2 : ResVGGBlock(64 → 128)  16×16 →  8×8
      Block3 : ResVGGBlock(128→ 256)   8×8  →  4×4
      Block4 : ResVGGBlock(256→ 512)   4×4  →  2×2
      GAP    : AdaptiveAvgPool2d → 512-dim vector
      Head   : Linear(512→512) → ReLU → Dropout(0.5) → Linear(512→62)
    """

    def __init__(self, num_classes: int = 62):
        super().__init__()

        self.block1 = _ResVGGBlock(  1,  64, dropout=0.10)   # 32×32 → 16×16
        self.block2 = _ResVGGBlock( 64, 128, dropout=0.10)   # 16×16 →  8×8
        self.block3 = _ResVGGBlock(128, 256, dropout=0.15)   #  8×8  →  4×4
        self.block4 = _ResVGGBlock(256, 512, dropout=0.15)   #  4×4  →  2×2

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
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
