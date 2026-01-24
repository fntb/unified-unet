# src/models/regressors.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from .wavelet_1d import haar_lowpass_multilevel

# 1) Backbone: Residual MLP (ResNet-style)

class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0, use_layernorm: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class MLPResNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        depth: int = 4,
        dropout: float = 0.0,
        use_layernorm: bool = False,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(hidden_dim, dropout=dropout, use_layernorm=use_layernorm) for _ in range(depth)]
        )
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        return self.head(h)



# 2) Simple preconditioners for 1D signals (vector form)

def moving_average_1d(y: torch.Tensor, kernel_size: int = 9) -> torch.Tensor:
    """
    y: (B, L) or (L,)
    returns: same shape as y
    """
    if y.dim() == 1:
        y_b = y[None, :]  # (1, L)
        squeeze = True
    else:
        y_b = y
        squeeze = False

    # Use conv1d: (B, 1, L)
    x = y_b.unsqueeze(1)
    k = kernel_size
    weight = torch.ones(1, 1, k, device=y.device, dtype=y.dtype) / float(k)
    pad = k // 2
    x_smooth = F.conv1d(F.pad(x, (pad, pad), mode="reflect"), weight)
    out = x_smooth.squeeze(1)

    return out.squeeze(0) if squeeze else out



# 3) Single configurable operator: baseline / residual / preconditioned
PrecondType = Literal["none", "identity", "ma", "haar"]

class DenoiserOperator(nn.Module):
    """
    One class that can emulate:
      - baseline:          Xhat = f(Y)
      - residual:          Xhat = Y + f(Y)
      - preconditioned:    Xhat = P(Y) + f(Y)
    where f is the same backbone (MLPResNet by default).
    """

    def __init__(self,
        net: nn.Module,
        mode: Literal["baseline", "residual", "preconditioned"] = "baseline",
        preconditioner: PrecondType = "none",
        ma_kernel_size: int = 9,
        haar_levels: int = 3,):
        super().__init__()
        self.net = net
        self.mode = mode
        self.preconditioner = preconditioner
        self.ma_kernel_size = int(ma_kernel_size)
        self.haar_levels = int(haar_levels)

    def P(self, y: torch.Tensor) -> torch.Tensor:
        if self.preconditioner == "none":
            # Only used in preconditioned mode, but keep safe default
            return torch.zeros_like(y)
        if self.preconditioner == "identity":
            return y
        if self.preconditioner == "ma":
            return moving_average_1d(y, kernel_size=self.ma_kernel_size)
        if self.preconditioner == "haar":
            return haar_lowpass_multilevel(y, levels=self.haar_levels)
        raise ValueError(f"Unknown preconditioner: {self.preconditioner}")

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if self.mode == "baseline":
            return self.net(y)

        if self.mode == "residual":
            return y + self.net(y)

        if self.mode == "preconditioned":
            p = self.P(y)
            return p + self.net(y)

        raise ValueError(f"Unknown mode: {self.mode}")