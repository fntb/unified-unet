# src/models/wavelet_1d.py
from __future__ import annotations

from typing import List, Tuple
import math
import torch


def _pad_to_even(x: torch.Tensor, mode: str = "reflect") -> Tuple[torch.Tensor, int]:
    """
    Pads last dimension by 1 if odd, returns (x_padded, pad_len)
    """
    L = x.shape[-1]
    if L % 2 == 0:
        return x, 0
    if mode == "reflect":
        pad_val = x[..., -2:-1]
    elif mode == "replicate":
        pad_val = x[..., -1:]
    else:
        raise ValueError("mode must be 'reflect' or 'replicate'")
    x = torch.cat([x, pad_val], dim=-1)
    return x, 1


def haar_dwt_1d(x: torch.Tensor, levels: int = 1, pad_mode: str = "reflect") -> Tuple[torch.Tensor, List[torch.Tensor], int]:
    """
    Orthonormal 1D Haar DWT.

    Args:
        x: (B, L) or (L,)
        levels: number of decomposition levels
    Returns:
        a: approximation at final level (B, L/2^levels)
        details: list [d1, d2, ..., d_levels], where d1 has length L/2, d2 has length L/4, etc.
        pad_len: 0 or 1 (if we padded to make length even at first level). We track it for perfect reconstruction.
    """
    squeeze = (x.dim() == 1)
    if squeeze:
        x = x[None, :]

    # For simplicity, pad only once at the start (if odd length)
    x, pad_len = _pad_to_even(x, mode=pad_mode)

    a = x
    details: List[torch.Tensor] = []
    inv_sqrt2 = 1.0 / math.sqrt(2.0)

    for _ in range(int(levels)):
        L = a.shape[-1]
        if L < 2:
            break
        if L % 2 == 1:
            # should not happen if we always keep even, but safe guard
            a, _ = _pad_to_even(a, mode=pad_mode)
            L = a.shape[-1]

        x_even = a[..., 0::2]
        x_odd  = a[..., 1::2]

        approx = (x_even + x_odd) * inv_sqrt2
        detail = (x_even - x_odd) * inv_sqrt2

        details.append(detail)
        a = approx

    return (a.squeeze(0) if squeeze else a,
            [d.squeeze(0) for d in details] if squeeze else details,
            pad_len)


def haar_idwt_1d(a: torch.Tensor, details: List[torch.Tensor], pad_len: int = 0) -> torch.Tensor:
    """
    Inverse orthonormal 1D Haar transform.

    Args:
        a: final approximation (B, L/2^levels) or (L/2^levels,)
        details: list [d1, d2, ..., d_levels] matching haar_dwt_1d output
        pad_len: 0 or 1 (crop at end)
    Returns:
        x reconstructed with original length (cropped if pad_len=1)
    """
    squeeze = (a.dim() == 1)
    if squeeze:
        a = a[None, :]
        details = [d[None, :] for d in details]

    inv_sqrt2 = 1.0 / math.sqrt(2.0)

    x = a
    for d in reversed(details):
        # x is approx at current level
        approx = x
        detail = d

        x_even = (approx + detail) * inv_sqrt2
        x_odd  = (approx - detail) * inv_sqrt2

        # interleave
        L = x_even.shape[-1]
        out = torch.empty(approx.shape[0], 2 * L, device=approx.device, dtype=approx.dtype)
        out[..., 0::2] = x_even
        out[..., 1::2] = x_odd
        x = out

    if pad_len == 1:
        x = x[..., :-1]

    return x.squeeze(0) if squeeze else x


def haar_lowpass_reconstruct(y: torch.Tensor, levels: int = 3, pad_mode: str = "reflect") -> torch.Tensor:
    """
    True Haar low-pass approximation at 'levels' obtained by:
    DWT -> zero all details -> IDWT.
    """
    a, details, pad_len = haar_dwt_1d(y, levels=levels, pad_mode=pad_mode)

    zero_details = [torch.zeros_like(d) for d in details]
    y_lp = haar_idwt_1d(a, zero_details, pad_len=pad_len)
    return y_lp