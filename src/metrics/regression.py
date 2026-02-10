# src/metrics/regression.py
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.metrics.frequency import spectral_mse_amplitude, bandwise_snr_db


# Core scalar metrics (batch-averaged)
def mse(xhat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.mean((xhat - x) ** 2)


def rmse(xhat: torch.Tensor, x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.sqrt(mse(xhat, x) + eps)


def mae(xhat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(xhat - x))


def huber(xhat: torch.Tensor, x: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    return F.smooth_l1_loss(xhat, x, beta=float(delta), reduction="mean")


def r2_score(xhat: torch.Tensor, x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    R2 = 1 - SSE/SST, computed per sample then averaged.
    """
    sse = torch.sum((x - xhat) ** 2, dim=1)
    x_mean = torch.mean(x, dim=1, keepdim=True)
    sst = torch.sum((x - x_mean) ** 2, dim=1) + eps
    return torch.mean(1.0 - sse / sst)


def corrcoef_batch(xhat: torch.Tensor, x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Pearson correlation per sample then averaged.
    """
    xhat_c = xhat - torch.mean(xhat, dim=1, keepdim=True)
    x_c = x - torch.mean(x, dim=1, keepdim=True)
    num = torch.sum(xhat_c * x_c, dim=1)
    den = torch.sqrt(torch.sum(xhat_c**2, dim=1) * torch.sum(x_c**2, dim=1) + eps)
    return torch.mean(num / den)


# Aliases (compat / readability)
corrcoef = corrcoef_batch  
spectral_mse_amp = spectral_mse_amplitude  


def snr_db(signal: torch.Tensor, noise: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    SNR in dB, averaged across batch.
    signal/noise: (B, L)
    """
    ps = torch.mean(signal**2, dim=1) + eps
    pn = torch.mean(noise**2, dim=1) + eps
    return torch.mean(10.0 * torch.log10(ps / pn))



# Aggregated regression metrics
@torch.no_grad()
def batch_regression_metrics(
    xhat: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor | None = None,
    huber_delta: float = 1.0,
) -> dict[str, float]:
    """
    Computes regression-like metrics between xhat and x.
    If y is provided (noisy input), adds copying indicators vs y.
    """
    if xhat.shape != x.shape:
        raise ValueError("xhat and x must have identical shapes")
    if xhat.dim() != 2:
        raise ValueError("expected tensors of shape (B, L)")

    eps = 1e-12

    out: dict[str, float] = {
        "mse_x": float(mse(xhat, x).item()),
        "rmse_x": float(rmse(xhat, x).item()),
        "mae_x": float(mae(xhat, x).item()),
        "huber_x": float(huber(xhat, x, delta=huber_delta).item()),
        "r2_x": float(r2_score(xhat, x).item()),
        "corr_x": float(corrcoef_batch(xhat, x).item()),
    }

    if y is not None:
        if y.shape != x.shape:
            raise ValueError("y must have same shape as x")
        out["mse_y"] = float(mse(xhat, y).item())
        out["mae_y"] = float(mae(xhat, y).item())
        out["copy_ratio_mse"] = float(out["mse_y"] / (out["mse_x"] + eps))

        # Reference noise level (input vs clean)
        ref_mse_yx = float(mse(y, x).item())
        ref_mae_yx = float(mae(y, x).item())
        out["ref_mse_yx"] = ref_mse_yx
        out["ref_mae_yx"] = ref_mae_yx

        # SNR (global)
        num = float(torch.mean(x**2).item()) + eps
        snr_in = 10.0 * np.log10(num / (ref_mse_yx + eps))
        snr_out = 10.0 * np.log10(num / (out["mse_x"] + eps))
        out["snr_in_db"] = float(snr_in)
        out["snr_out_db"] = float(snr_out)
        out["snr_gain_db"] = float(snr_out - snr_in)

    return out



# Multi-model evaluation
@torch.no_grad()
def evaluate_models(
    models: Dict[str, torch.nn.Module],
    Y: torch.Tensor,
    X: torch.Tensor,
    huber_delta: float = 1.0,
    lf_levels: int = 3,  # not used directly here, kept for API compatibility
    f_split: float = 0.15,
) -> pd.DataFrame:
    """
    Evaluates multiple models on (Y -> X) denoising.
    Returns a DataFrame sorted by mse_x.
    """
    if Y.shape != X.shape or Y.dim() != 2:
        raise ValueError("Y and X must be (B, L) with identical shapes")

    rows = []
    for name, model in models.items():
        xhat = model(Y)

        d = batch_regression_metrics(xhat=xhat, x=X, y=Y, huber_delta=huber_delta)

        # Frequency-domain metrics
        d["spec_mse_amp"] = float(spectral_mse_amplitude(xhat, X))
        snr_lf, snr_hf = bandwise_snr_db(xhat, X, f_split=f_split)
        d["snr_lf_db"] = float(snr_lf)
        d["snr_hf_db"] = float(snr_hf)

        rows.append({"name": name, **d})

    df = pd.DataFrame(rows).sort_values("mse_x").reset_index(drop=True)
    return df