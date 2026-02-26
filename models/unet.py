"""
U-Net implementation for binary brain-tumor segmentation.

Architecture follows the original Ronneberger et al. (2015) paper with
modern additions: BatchNorm, dropout in bottleneck, and flexible depth.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """Two consecutive Conv → BN → ReLU blocks."""

    def __init__(self, in_ch: int, out_ch: int, mid_ch: int = None,
                 dropout: float = 0.0):
        super().__init__()
        mid_ch = mid_ch or out_ch
        layers = [
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    """Max-pool then DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, dropout=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class Up(nn.Module):
    """Upsample then DoubleConv (with skip connection cat)."""

    def __init__(self, in_ch: int, out_ch: int,
                 bilinear: bool = True, dropout: float = 0.0):
        super().__init__()
        if bilinear:
            self.up   = nn.Upsample(scale_factor=2, mode="bilinear",
                                    align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, in_ch // 2,
                                   dropout=dropout)
        else:
            self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2,
                                           kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # handle odd spatial sizes
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        x  = F.pad(x, [dx // 2, dx - dx // 2,
                        dy // 2, dy - dy // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class OutConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """
    Flexible U-Net.

    Parameters
    ----------
    in_channels  : number of input channels (1 for grayscale MRI)
    out_channels : number of output channels (1 for binary segmentation)
    features     : channel counts at each encoder depth
    bilinear     : use bilinear upsampling (True) or transposed conv (False)
    dropout      : dropout rate applied in bottleneck
    """

    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 features: list = None,
                 bilinear: bool = True,
                 dropout: float = 0.3):
        super().__init__()
        features = features or [64, 128, 256, 512]

        self.inc  = DoubleConv(in_channels, features[0])
        self.downs = nn.ModuleList()
        self.ups   = nn.ModuleList()

        # encoder
        for i in range(len(features) - 1):
            self.downs.append(Down(features[i], features[i + 1]))

        # bottleneck
        factor = 2 if bilinear else 1
        self.bottleneck = Down(features[-1],
                               features[-1] * 2 // factor,
                               dropout=dropout)

        # decoder
        dec_features = [features[-1] * 2] + list(reversed(features))
        for i in range(len(features)):
            self.ups.append(
                Up(dec_features[i], dec_features[i + 1] // factor,
                   bilinear=bilinear)
            )

        self.outc = OutConv(features[0] // factor if bilinear else features[0],
                            out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [self.inc(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))

        x = self.bottleneck(skips[-1])

        for i, up in enumerate(self.ups):
            x = up(x, skips[-(i + 1)])

        return self.outc(x)  # raw logits


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = UNet(in_channels=1, out_channels=1)
    dummy = torch.randn(2, 1, 256, 256)
    out   = model(dummy)
    print(f"Input : {dummy.shape}")
    print(f"Output: {out.shape}")  # (2, 1, 256, 256)

    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")
