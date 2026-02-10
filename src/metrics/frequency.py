# src/metrics/frequency.py
from __future__ import annotations

import torch


def rfft_amplitude(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B, L) -> |rfft(x)|: (B, L//2+1)
    """
    if x.dim() != 2:
        raise ValueError("expected x of shape (B, L)")
    return torch.abs(torch.fft.rfft(x, dim=1))


@torch.no_grad()
def spectral_mse_amplitude(xhat: torch.Tensor, x: torch.Tensor) -> float:
    """
    MSE between magnitude spectra |F(xhat)| and |F(x)|.
    """
    if xhat.shape != x.shape:
        raise ValueError("xhat and x must have the same shape")
    a_hat = rfft_amplitude(xhat)
    a = rfft_amplitude(x)
    return float(torch.mean((a_hat - a) ** 2).item())


def _band_masks(L: int, fs: float = 1.0, f_split: float = 0.15) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Builds boolean masks over rFFT bins for LF/HF split.

    rfft bins: k=0..L//2, freq = k*fs/L, Nyquist=fs/2.
    f_split is a fraction of Nyquist (e.g. 0.15 means 15% of Nyquist).
    """
    if not (0.0 < f_split < 1.0):
        raise ValueError("f_split must be in (0, 1)")

    freqs = torch.linspace(0.0, 0.5 * fs, L // 2 + 1)
    f_cut = f_split * (0.5 * fs)
    lf = freqs <= f_cut
    hf = freqs > f_cut
    return lf, hf


@torch.no_grad()
def bandwise_snr_db(
    xhat: torch.Tensor,
    x: torch.Tensor,
    f_split: float = 0.15,
    fs: float = 1.0,
    eps: float = 1e-12,
) -> tuple[float, float]:
    """
    Bandwise SNR (LF and HF) in the frequency domain using amplitude^2 as power proxy.

    signal: x
    noise:  (x - xhat)

    Returns:
        (snr_lf_db, snr_hf_db)
    """
    if xhat.shape != x.shape or x.dim() != 2:
        raise ValueError("xhat and x must be (B, L) with same shape")

    _, L = x.shape
    lf_mask, hf_mask = _band_masks(L, fs=fs, f_split=f_split)
    lf_mask = lf_mask.to(x.device)
    hf_mask = hf_mask.to(x.device)

    A_x = rfft_amplitude(x)
    A_e = rfft_amplitude(x - xhat)

    ps_lf = torch.mean(A_x[:, lf_mask] ** 2) + eps
    pn_lf = torch.mean(A_e[:, lf_mask] ** 2) + eps
    ps_hf = torch.mean(A_x[:, hf_mask] ** 2) + eps
    pn_hf = torch.mean(A_e[:, hf_mask] ** 2) + eps

    snr_lf = 10.0 * torch.log10(ps_lf / pn_lf)
    snr_hf = 10.0 * torch.log10(ps_hf / pn_hf)
    return float(snr_lf.item()), float(snr_hf.item())


# Alias pratique (si tu veux un nom court)
spectral_mse_amp = spectral_mse_amplitude