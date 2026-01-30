# src/analysis/reconstruction_analysis.py
from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from src.data.signal_1d import Signal1DConfig, generate_pair
from src.metrics.regression import evaluate_models
from src.metrics.wavelet import mse_lf_hf, nmse_lf_hf  


# Fixed batch utilities
def make_fixed_batch(
    cfg: Signal1DConfig,
    seeds: List[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        Y: noisy inputs  (B, L)
        X: clean targets (B, L)
    """
    xs, ys = [], []
    for s in seeds:
        x, y = generate_pair(cfg, seed=int(s), device=torch.device("cpu"), return_components=False)
        xs.append(x)
        ys.append(y)
    X = torch.stack(xs, dim=0).to(device)
    Y = torch.stack(ys, dim=0).to(device)
    return Y, X



# Qualitative plots
@torch.no_grad()
def plot_reconstructions(
    models: Dict[str, torch.nn.Module],
    Y: torch.Tensor,
    X: torch.Tensor,
    seeds: List[int],
    max_models: int | None = None,
):
    """
    For each sample i, plots:
      - Input (X clean vs Y noisy)
      - One row per model: X vs X_hat
    """
    if Y.shape != X.shape or Y.dim() != 2:
        raise ValueError("Y and X must be (B, L) with identical shapes")

    names = list(models.keys())
    if max_models is not None:
        names = names[: int(max_models)]

    B, L = X.shape
    t = np.arange(L)

    for i in range(B):
        fig, axes = plt.subplots(
            1 + len(names),
            1,
            figsize=(12, 2.25 * (1 + len(names))),
            sharex=True,
        )

        axes[0].plot(t, X[i].detach().cpu().numpy(), label="X (clean)", linewidth=2)
        axes[0].plot(t, Y[i].detach().cpu().numpy(), label="Y (noisy)", alpha=0.6)
        axes[0].set_title(f"Seed={seeds[i]} | Input")
        axes[0].legend()
        axes[0].grid(alpha=0.2)

        for k, name in enumerate(names, start=1):
            m = models[name]
            xhat = m(Y[i : i + 1]).squeeze(0)

            axes[k].plot(t, X[i].detach().cpu().numpy(), label="X", linewidth=2)
            axes[k].plot(t, xhat.detach().cpu().numpy(), label="X_hat", alpha=0.85)
            axes[k].set_title(name)
            axes[k].legend()
            axes[k].grid(alpha=0.2)

        plt.tight_layout()
        plt.show()


# Quantitative evaluation
@torch.no_grad()
def eval_models_table(
    models: Dict[str, torch.nn.Module],
    Y: torch.Tensor,
    X: torch.Tensor,
    lf_levels: int = 3,
    huber_delta: float = 1.0,
    f_split: float = 0.15,
) -> pd.DataFrame:
    """
    Builds a nice metrics table for multiple models.
    """
    df = evaluate_models(
        models=models,
        Y=Y,
        X=X,
        huber_delta=huber_delta,
        lf_levels=lf_levels,
        f_split=f_split,
    )

    # Add LF/HF (Haar) breakdown (raw + normalized)
    lf_rows = []
    for name in df["name"].tolist():
        xhat = models[name](Y)
        d1 = mse_lf_hf(xhat, X, levels=lf_levels)
        d2 = nmse_lf_hf(xhat, X, levels=lf_levels)
        lf_rows.append({**d1, **d2})

    df_lf = pd.DataFrame(lf_rows)
    df = pd.concat([df, df_lf], axis=1)

    # Order columns (feel free to adjust)
    col_order = [
        "name",
        "mse_x",
        "mae_x",
        "huber_x",
        "r2_x",
        "corr_x",
        "mse_y",
        "mae_y",
        "copy_ratio_mse",
        "snr_in_db",
        "snr_out_db",
        "snr_gain_db",
        "spec_mse_amp",
        "snr_lf_db",
        "snr_hf_db",
        "mse",
        "mse_lf",
        "mse_hf",
        "nmse_lf",
        "nmse_hf",
        "ref_mse_yx",
        "ref_mae_yx",
    ]
    col_order = [c for c in col_order if c in df.columns]
    df = df[col_order].sort_values("mse_x").reset_index(drop=True)
    return df


def export_metrics_excel(df: pd.DataFrame, out_path: str) -> None:
    out_path = str(out_path)
    df.to_excel(out_path, index=False)



# Small plotting helper
def plot_metric_lines(df: pd.DataFrame, cols: list[str], title: str = "") -> None:
    x = np.arange(len(df))
    plt.figure(figsize=(11, 4))
    for c in cols:
        if c not in df.columns:
            continue
        plt.plot(x, df[c].values, marker="o", label=c)
    plt.xticks(x, df["name"].values, rotation=15, ha="right")
    plt.title(title)
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()
