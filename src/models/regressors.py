# src/models/regressors.py
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wavelet_1d import haar_lowpass_reconstruct

# 1) Backbone: simple Residual MLP (vector-to-vector)
class ResidualMLPBlock(nn.Module):
    """
    Residual block on vectors: x -> x + g(LN(x))
    """
    def __init__(self, dim: int, dropout: float = 0.0, use_layernorm: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(dim) if use_layernorm else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class MLPResNet(nn.Module):
    """
    Small residual MLP for denoising/regression on 1D signals in vector form.

    Expects:
        input:  (B, L)
        output: (B, L)
    """
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
        if input_dim != output_dim:
            raise ValueError(f"MLPResNet expects input_dim == output_dim, got {input_dim} != {output_dim}")

        self.L = int(input_dim)

        self.stem = nn.Sequential(
            nn.Linear(self.L, hidden_dim),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(hidden_dim, dropout=dropout, use_layernorm=use_layernorm) for _ in range(int(depth))]
        )
        self.head = nn.Linear(hidden_dim, self.L)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"MLPResNet expects shape (B, L), got {tuple(x.shape)}")
        h = self.stem(x)
        for blk in self.blocks:
            h = blk(h)
        return self.head(h)



# 2) Simple 1D preconditioners (work on (B,L) or (L,))
def moving_average_1d(y: torch.Tensor, kernel_size: int = 9) -> torch.Tensor:
    """
    Simple moving average smoothing (reflection padding).

    Args:
        y: (B, L) or (L,)
        kernel_size: odd integer recommended

    Returns:
        same shape as y
    """
    if kernel_size <= 1:
        return y

    squeeze = (y.dim() == 1)
    if squeeze:
        y = y[None, :]  # (1, L)

    # conv1d expects (B, C, L)
    x = y.unsqueeze(1)
    k = int(kernel_size)
    if (k % 2) == 0:
        raise ValueError("moving_average_1d expects an odd kernel_size for symmetric padding.")

    weight = torch.ones(1, 1, k, device=y.device, dtype=y.dtype) / float(k)
    pad = k // 2
    x_smooth = F.conv1d(F.pad(x, (pad, pad), mode="reflect"), weight)
    out = x_smooth.squeeze(1)

    return out.squeeze(0) if squeeze else out


# 3) Denoiser operator: baseline / residual / preconditioned
PrecondType = Literal["identity", "ma", "haar"]
ModeType = Literal["baseline", "residual", "preconditioned"]


class DenoiserOperator(nn.Module):
    """
    Wrap a backbone network f into one of the following operators:

    - baseline:
        x_hat = f(y)

    - residual:
        x_hat = y + f(y)

    - preconditioned:
        x_hat = p(y) + f(p(y))

    This matches the "learn a correction around a coarse approximation" idea.
    """

    def __init__(
        self,
        net: nn.Module,
        mode: ModeType = "baseline",
        preconditioner: PrecondType = "identity",
        ma_kernel_size: int = 9,
        haar_levels: int = 3,
    ):
        super().__init__()
        self.net = net
        self.mode = mode
        self.preconditioner = preconditioner
        self.ma_kernel_size = int(ma_kernel_size)
        self.haar_levels = int(haar_levels)

        if self.mode == "preconditioned" and self.preconditioner is None:
            raise ValueError("preconditioned mode requires a valid preconditioner.")

    def P(self, y: torch.Tensor) -> torch.Tensor:
        if self.preconditioner == "identity":
            return y
        if self.preconditioner == "ma":
            return moving_average_1d(y, kernel_size=self.ma_kernel_size)
        if self.preconditioner == "haar":
            return haar_lowpass_reconstruct(y, levels=self.haar_levels)
        raise ValueError(f"Unknown preconditioner: {self.preconditioner}")

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if y.dim() != 2:
            raise ValueError(f"DenoiserOperator expects input shape (B, L), got {tuple(y.shape)}")

        if self.mode == "baseline":
            return self.net(y)

        if self.mode == "residual":
            return y + self.net(y)

        if self.mode == "preconditioned":
            p = self.P(y)
            return p + self.net(y-p)

        raise ValueError(f"Unknown mode: {self.mode}")