# src/models/regression_model.py
from __future__ import annotations

import torch
from .model import Model


class RegressionModel(Model):
    """
    LightningModule for denoising/regression.

    Expected batch format:
        (y_noisy, x_clean)
    """

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        y_noisy, x_clean = batch
        x_hat = self(y_noisy)
        loss = self.loss(x_hat, x_clean)
        self.log(f"{stage}_loss", loss, prog_bar=(stage != "test"))
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        y_noisy = batch[0] if isinstance(batch, (tuple, list)) else batch
        x_hat = self(y_noisy)
        return x_hat