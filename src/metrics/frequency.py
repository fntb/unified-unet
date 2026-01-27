import torch
from src.models.wavelet_1d import haar_lowpass_reconstruct

def split_lf_hf(x: torch.Tensor, levels: int = 3):
    x_lf = haar_lowpass_reconstruct(x, levels=levels)
    x_hf = x - x_lf
    return x_lf, x_hf


@torch.no_grad()
def mse_lf_hf(x_hat: torch.Tensor, x: torch.Tensor, levels: int = 3):
    x_lf, x_hf = split_lf_hf(x, levels)
    x_hat_lf, x_hat_hf = split_lf_hf(x_hat, levels)

    return {
        "mse": torch.mean((x_hat - x) ** 2).item(),
        "mse_lf": torch.mean((x_hat_lf - x_lf) ** 2).item(),
        "mse_hf": torch.mean((x_hat_hf - x_hf) ** 2).item(),
    }