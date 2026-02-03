# src/metrics/wavelet.py
from __future__ import annotations

import torch

from src.models.wavelet_1d import haar_lowpass_reconstruct


def split_lf_hf(x: torch.Tensor, levels: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Split x into:
      - x_lf: Haar low-pass reconstruction at 'levels'
      - x_hf: residual (high-frequency details)
    Works with x shaped (B, L) or (L,).
    """
    x_lf = haar_lowpass_reconstruct(x, levels=levels)
    x_hf = x - x_lf
    return x_lf, x_hf


@torch.no_grad()
def mse_lf_hf(x_hat: torch.Tensor, x: torch.Tensor, levels: int = 3) -> dict[str, float]:
    """
    Returns batch-averaged MSE on:
      - full signal
      - low-frequency part
      - high-frequency part
    """
    x_lf, x_hf = split_lf_hf(x, levels=levels)
    x_hat_lf, x_hat_hf = split_lf_hf(x_hat, levels=levels)

    return {
        "mse": float(torch.mean((x_hat - x) ** 2).item()),
        "mse_lf": float(torch.mean((x_hat_lf - x_lf) ** 2).item()),
        "mse_hf": float(torch.mean((x_hat_hf - x_hf) ** 2).item()),
    }


@torch.no_grad()
def nmse_lf_hf(x_hat: torch.Tensor, x: torch.Tensor, levels: int = 3, eps: float = 1e-12) -> dict[str, float]:
    """
    Normalized MSE, to compare LF vs HF on a fair scale:
      NMSE_LF = MSE_LF / E[ x_LF^2 ]
      NMSE_HF = MSE_HF / E[ x_HF^2 ]
    """
    x_lf, x_hf = split_lf_hf(x, levels=levels)
    x_hat_lf, x_hat_hf = split_lf_hf(x_hat, levels=levels)

    mse_lf = torch.mean((x_hat_lf - x_lf) ** 2)
    mse_hf = torch.mean((x_hat_hf - x_hf) ** 2)

    denom_lf = torch.mean(x_lf ** 2) + eps
    denom_hf = torch.mean(x_hf ** 2) + eps

    return {
        "nmse_lf": float((mse_lf / denom_lf).item()),
        "nmse_hf": float((mse_hf / denom_hf).item()),
    }