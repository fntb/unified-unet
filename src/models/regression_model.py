# src/models/regression_model.py

import torch
from .model import Model


class RegressionModel(Model):
    """
    LightningModule for regression / denoising problems.

    Assumes batches are tuples (Y, X):
      - Y: noisy observation (input)
      - X: clean signal (target)
    """

    def training_step(self, batch, batch_idx):
        y_noisy, x_clean = batch
        x_hat = self(y_noisy)

        loss = self.loss(x_hat, x_clean)
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        y_noisy, x_clean = batch
        x_hat = self(y_noisy)

        loss = self.loss(x_hat, x_clean)
        self.log("val_loss", loss, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        y_noisy, x_clean = batch
        x_hat = self(y_noisy)

        loss = self.loss(x_hat, x_clean)
        self.log("test_loss", loss)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        For prediction, we only care about the reconstructed signal.
        """
        y_noisy = batch[0] if isinstance(batch, (tuple, list)) else batch
        x_hat = self(y_noisy)
        return x_hat