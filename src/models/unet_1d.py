# src/models/unet_1d.py
from __future__ import annotations

from typing import List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv1d(nn.Module):
    """(Conv1d -> BN -> ReLU) x2."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class UNetDownBlock1d(nn.Module):
    """DoubleConv then downsample (maxpool). Returns (x_down, x_skip)."""
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = DoubleConv1d(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(x)
        x_skip = x
        x_down = F.max_pool1d(x, kernel_size=2, stride=2)
        return x_down, x_skip


class UNetUpBlock1d(nn.Module):
    """
    Upsample then concat skip then DoubleConv.
    We use linear upsample + 1x1 conv like your 2D version.
    """
    def __init__(self, in_ch: int, out_ch: int, skip_ch: int) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=True),
            nn.Conv1d(in_ch, in_ch - skip_ch, kernel_size=1),
        )
        self.conv = DoubleConv1d(in_ch, out_ch)

    def forward(self, x: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # Handle odd lengths due to pooling/cropping
        if x.shape[-1] != x_skip.shape[-1]:
            diff = x_skip.shape[-1] - x.shape[-1]
            if diff > 0:
                x = F.pad(x, (diff // 2, diff - diff // 2))
            else:
                x = x[..., (-diff)//2 : x.shape[-1] + diff//2]  # rare case: crop

        x = torch.cat([x_skip, x], dim=1)
        x = self.conv(x)
        return x


class UNet1D(nn.Module):
    """
    U-Net 1D for denoising/regression.
    Input:  (B, L) or (B, 1, L)
    Output: (B, L)

    skip_mode:
      - "on": standard U-Net
      - "off": ablation, skips replaced by zeros (same shape)
    """
    def __init__(
        self,
        in_channels: int = 1,
        dims: List[int] = [32, 64, 128, 256],
        out_channels: int = 1,
        skip_mode: Literal["on", "off"] = "on",
    ) -> None:
        super().__init__()
        self.skip_mode = skip_mode

        # Encoder
        self.encoder = nn.ModuleList()
        chs = [in_channels] + dims
        for i in range(len(dims)):
            self.encoder.append(UNetDownBlock1d(chs[i], chs[i+1]))

        # Bottleneck
        self.bottleneck = DoubleConv1d(dims[-1], dims[-1] * 2)
        bottleneck_ch = dims[-1] * 2

        # Decoder
        self.decoder = nn.ModuleList()
        rev_dims = list(reversed(dims))
        in_ch = bottleneck_ch
        for out_ch, skip_ch in zip(rev_dims, rev_dims):
            self.decoder.append(UNetUpBlock1d(in_ch, out_ch, skip_ch=skip_ch))
            in_ch = out_ch

        # Head
        self.head = nn.Conv1d(dims[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (B,L) or (B,1,L)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        skips: list[torch.Tensor] = []
        h = x
        for enc in self.encoder:
            h, h_skip = enc(h)
            skips.append(h_skip)

        h = self.bottleneck(h)

        for dec, h_skip in zip(self.decoder, reversed(skips)):
            if self.skip_mode == "off":
                h_skip = torch.zeros_like(h_skip)
            h = dec(h, h_skip)

        out = self.head(h)  # (B,1,L)
        return out.squeeze(1)