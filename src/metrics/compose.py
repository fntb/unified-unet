from typing import (
    Iterable,
    Tuple
)

import torch
import torch.nn as nn

class ComposeLoss(nn.Module):
    def __init__(self, losses: Iterable[Tuple[nn.Module, float]]) -> None:
        super().__init__()

        self.weights = [weight for _, weight in losses]
        self.losses = nn.ModuleList([loss for loss, _ in losses])

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor):
        
        combine_loss = torch.tensor(0., device=y_hat.device, dtype=y_hat.dtype)

        for weight, loss in zip(self.weights, self.losses):
            combine_loss += weight * loss(y_hat, y)

        return combine_loss

