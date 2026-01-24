# src/models/wavelet_1d.py
from __future__ import annotations

import torch
import torch.nn.functional as F


def haar_lowpass_multilevel(y: torch.Tensor, levels: int = 3) -> torch.Tensor:
    """
    Multi-level Haar-style low-pass approximation for 1D signals.

    Downsample: block averaging (approximation coefficients).
    Upsample: nearest (repeat) to preserve alignment and avoid interpolation artefacts.

    Args:
        y: (B, L) or (L,)
        levels: number of pooling (downsampling) levels

    Returns:
        same shape as y
    """
    squeeze = (y.dim() == 1)
    if squeeze:
        y = y[None, :]  # (1, L)

    B, L = y.shape
    x = y.unsqueeze(1)  # (B, 1, L)

    applied_levels = 0

    # Downsample: block averaging
    for _ in range(int(levels)):
        if x.shape[-1] < 2:
            break

        # If odd length, pad one sample (reflect) to make pooling safe
        if (x.shape[-1] % 2) == 1:
            x = F.pad(x, (0, 1), mode="reflect")

        x = F.avg_pool1d(x, kernel_size=2, stride=2)
        applied_levels += 1

    # Upsample back by repeat (nearest)
    up_factor = 2 ** applied_levels
    x_up = x.repeat_interleave(up_factor, dim=-1)

    # Crop/pad to original length L
    if x_up.shape[-1] > L:
        x_up = x_up[..., :L]
    elif x_up.shape[-1] < L:
        # replicate tends to match "nearest" spirit; reflect is also ok
        x_up = F.pad(x_up, (0, L - x_up.shape[-1]), mode="replicate")

    out = x_up.squeeze(1)  # (B, L)
    return out.squeeze(0) if squeeze else out