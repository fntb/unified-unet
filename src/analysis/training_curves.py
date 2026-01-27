import os
import pandas as pd
import matplotlib.pyplot as plt


def read_metrics_csv(version: str, log_dir: str) -> pd.DataFrame:
    path = os.path.join(log_dir, version, "metrics.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"metrics.csv not found: {path}")
    return pd.read_csv(path)


def _pick_metric_column(df: pd.DataFrame, base_name: str) -> str | None:
    for c in (f"{base_name}_epoch", base_name):
        if c in df.columns:
            return c
    return None


def curve_by_epoch(df: pd.DataFrame, metric_base_name: str) -> pd.DataFrame | None:
    if "epoch" not in df.columns:
        return None
    col = _pick_metric_column(df, metric_base_name)
    if col is None:
        return None

    d = df.dropna(subset=["epoch", col]).copy()
    if len(d) == 0:
        return None

    d["epoch"] = d["epoch"].astype(int)

    if "step" in d.columns:
        d = d.sort_values(["epoch", "step"]).groupby("epoch", as_index=False).tail(1)
    else:
        d = d.sort_values(["epoch"]).groupby("epoch", as_index=False).tail(1)

    return d[["epoch", col]].rename(columns={col: metric_base_name}).sort_values("epoch").reset_index(drop=True)


def plot_train_val_per_model(run_version: dict, log_dir: str):
    for name, version in run_version.items():
        df = read_metrics_csv(version, log_dir)
        train = curve_by_epoch(df, "train_loss")
        val = curve_by_epoch(df, "val_loss")

        if train is None and val is None:
            print(f"[WARN] {name} ({version}): no train/val loss found. columns={list(df.columns)}")
            continue

        plt.figure(figsize=(9, 4))
        if train is not None:
            plt.plot(train["epoch"], train["train_loss"], label="train_loss")
        if val is not None:
            plt.plot(val["epoch"], val["val_loss"], label="val_loss")

        plt.title(f"{name} | train vs val (per epoch)")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.grid(alpha=0.2)
        plt.legend()
        plt.tight_layout()
        plt.show()


def plot_val_loss_overlay(run_version: dict, log_dir: str):
    plt.figure(figsize=(9, 4))
    plotted = 0

    for name, version in run_version.items():
        df = read_metrics_csv(version, log_dir)
        val = curve_by_epoch(df, "val_loss")
        if val is None:
            print(f"[WARN] {name} ({version}): no val_loss found.")
            continue
        plt.plot(val["epoch"], val["val_loss"], label=name)
        plotted += 1

    if plotted == 0:
        print("[ERROR] No val_loss curves found.")
        return

    plt.title("Validation loss comparison (same scale)")
    plt.xlabel("epoch")
    plt.ylabel("val_loss")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()