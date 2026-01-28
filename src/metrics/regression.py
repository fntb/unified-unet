import torch

def mse(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.mean((x_hat - x) ** 2)