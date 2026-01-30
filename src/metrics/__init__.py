# src/metrics/__init__.py
from .accuracy import accuracy, Accuracy
from .dice import DiceLoss, MulticlassDiceLoss, SingleclassDiceLoss
from .compose import ComposeLoss

# Regression metrics
from .regression import (
    mse,
    mae,
    huber,
    r2_score,
    corrcoef_batch,
    corrcoef,        # alias = corrcoef_batch
    snr_db,
)

# Frequency metrics
from .frequency import (
    spectral_mse_amplitude,
    spectral_mse_amp,     # alias
    bandwise_snr_db,
)

# Wavelet / multi-scale metrics
from .wavelet import (
    split_lf_hf,
    mse_lf_hf,
    nmse_lf_hf,
)

__all__ = [
    # classification / seg
    "accuracy",
    "Accuracy",
    "DiceLoss",
    "MulticlassDiceLoss",
    "SingleclassDiceLoss",

    # loss composition
    "ComposeLoss",

    # regression
    "mse",
    "mae",
    "huber",
    "r2_score",
    "corrcoef_batch",
    "corrcoef",
    "snr_db",

    # frequency
    "spectral_mse_amplitude",
    "spectral_mse_amp",
    "bandwise_snr_db",

    # wavelet
    "split_lf_hf",
    "mse_lf_hf",
    "nmse_lf_hf",
]