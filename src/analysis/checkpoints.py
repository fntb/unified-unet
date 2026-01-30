import os
import glob
import re
import pandas as pd
import torch

from src.models.regression_model import RegressionModel


def read_metrics_csv(version: str, log_dir: str) -> pd.DataFrame:
    path = os.path.join(log_dir, version, "metrics.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing metrics.csv: {path}")
    return pd.read_csv(path)


def best_epoch_and_val(version: str, log_dir: str) -> tuple[int, float]:
    df = read_metrics_csv(version, log_dir)
    d = df.dropna(subset=["epoch", "val_loss"]).copy()
    if len(d) == 0:
        raise ValueError(f"No val_loss found in {version}/metrics.csv")
    d["epoch"] = d["epoch"].astype(int)
    best = d.sort_values(["val_loss", "epoch"], ascending=[True, True]).iloc[0]
    return int(best["epoch"]), float(best["val_loss"])


def _parse_val_loss_from_ckpt_name(path: str) -> float:
    m = re.search(r"val_loss=([0-9]+(?:\.[0-9]+)?)", os.path.basename(path))
    return float(m.group(1)) if m else float("inf")


def find_ckpt_for_best(version: str, ckpt_dir: str, log_dir: str) -> tuple[str, int, float]:
    best_epoch, best_val = best_epoch_and_val(version, log_dir)

    # Try exact 6-decimal match first (Lightning filename formatting)
    val6 = f"{best_val:.6f}"
    exact = glob.glob(os.path.join(ckpt_dir, f"*epoch={best_epoch}*-val_loss={val6}*.ckpt"))
    if len(exact) >= 1:
        exact.sort(key=os.path.getmtime, reverse=True)
        return exact[0], best_epoch, best_val

    # Fallback: match epoch only, pick closest val_loss parsed from filename
    cands = glob.glob(os.path.join(ckpt_dir, f"*epoch={best_epoch}*-val_loss=*.ckpt"))
    if len(cands) == 0:
        raise FileNotFoundError(f"No checkpoint found for epoch={best_epoch} in {ckpt_dir}")

    cands.sort(key=lambda p: abs(_parse_val_loss_from_ckpt_name(p) - best_val))
    return cands[0], best_epoch, best_val


def load_best_models(
    run_version: dict,
    ckpt_dir: str,
    log_dir: str,
    device: torch.device,
) -> dict:
    """
    Returns:
      {
        name: {
          "model": LightningModule,
          "ckpt": str,
          "best_epoch": int,
          "best_val": float
        }
      }
    """
    out = {}

    for name, version in run_version.items():
        ckpt, ep, best_val = find_ckpt_for_best(version, ckpt_dir, log_dir)

        # PyTorch 2.6 / Lightning: checkpoints may include DictConfig, so weights_only must be False.
        model = RegressionModel.load_from_checkpoint(
            ckpt,
            map_location=device,
            weights_only=False,
        ).to(device).eval()

        out[name] = {
            "model": model,
            "ckpt": ckpt,
            "best_epoch": ep,
            "best_val": best_val,
        }

    return out
