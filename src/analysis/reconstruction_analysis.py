import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from src.data.signal_1d import Signal1DConfig, generate_pair
from src.models.wavelet_1d import haar_lowpass_reconstruct


def fixed_batch(cfg: Signal1DConfig, seeds: list[int], device: torch.device):
    xs, ys = [], []
    for s in seeds:
        x, y = generate_pair(cfg, seed=int(s), device=torch.device("cpu"))
        xs.append(x)
        ys.append(y)
    X = torch.stack(xs, dim=0).to(device)  # clean
    Y = torch.stack(ys, dim=0).to(device)  # noisy
    return Y, X


@torch.no_grad()
def plot_recons_clean(models: dict, Y: torch.Tensor, X: torch.Tensor, seeds: list[int]):
    B, L = X.shape
    t = np.arange(L)

    for i in range(B):
        fig, axes = plt.subplots(
            1 + len(models),
            1,
            figsize=(12, 2.3 * (1 + len(models))),
            sharex=True,
        )

        axes[0].plot(t, X[i].detach().cpu().numpy(), label="X (clean)", linewidth=2)
        axes[0].plot(t, Y[i].detach().cpu().numpy(), label="Y (noisy)", alpha=0.6)
        axes[0].set_title(f"Seed={seeds[i]} | Input")
        axes[0].legend()
        axes[0].grid(alpha=0.2)

        for k, (name, m) in enumerate(models.items(), start=1):
            xhat = m(Y[i:i+1]).squeeze(0)
            axes[k].plot(t, X[i].detach().cpu().numpy(), label="X", linewidth=2)
            axes[k].plot(t, xhat.detach().cpu().numpy(), label="X_hat", alpha=0.85)
            axes[k].set_title(name)
            axes[k].legend()
            axes[k].grid(alpha=0.2)

        plt.tight_layout()
        plt.show()


def lf_hf(x: torch.Tensor, levels: int = 3):
    x_lf = haar_lowpass_reconstruct(x, levels=levels)
    x_hf = x - x_lf
    return x_lf, x_hf


@torch.no_grad()
def eval_models_lf_hf(models: dict, Y: torch.Tensor, X: torch.Tensor, lf_levels: int = 3) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        Xhat = model(Y)

        mse = torch.mean((Xhat - X) ** 2).item()

        X_lf, X_hf = lf_hf(X, levels=lf_levels)
        Xhat_lf, Xhat_hf = lf_hf(Xhat, levels=lf_levels)

        mse_lf = torch.mean((Xhat_lf - X_lf) ** 2).item()
        mse_hf = torch.mean((Xhat_hf - X_hf) ** 2).item()

        rows.append({"name": name, "mse": mse, "mse_lf": mse_lf, "mse_hf": mse_hf})

    return pd.DataFrame(rows).sort_values("mse").reset_index(drop=True)


def plot_lf_hf_bars(df: pd.DataFrame, title: str):
    df = df.copy().reset_index(drop=True)
    x = np.arange(len(df))
    w = 0.25

    plt.figure(figsize=(12, 4))
    plt.bar(x - w, df["mse"].values, w, label="MSE total")
    plt.bar(x,     df["mse_lf"].values, w, label="MSE LF")
    plt.bar(x + w, df["mse_hf"].values, w, label="MSE HF")
    plt.xticks(x, df["name"].values, rotation=25, ha="right")
    plt.ylabel("MSE")
    plt.title(title)
    plt.grid(axis="y", alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()