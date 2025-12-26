from typing import (
    Optional
)

import torch
import torch.nn as nn
import pytorch_lightning as pl
from hydra.utils import (
    instantiate,
)
import omegaconf

class Model(pl.LightningModule):
    def __init__(
        self, 
        model: omegaconf.DictConfig,
        loss: omegaconf.DictConfig,
        optimizer: Optional[omegaconf.DictConfig] = None,
        scheduler: Optional[omegaconf.DictConfig] = None,
        metrics: Optional[omegaconf.DictConfig] = None
    ):
        super().__init__()

        self.save_hyperparameters("model", "optimizer", "scheduler", "loss")
        self.optimizer_conf = optimizer
        self.scheduler_conf = scheduler

        self.model = instantiate(model)
        self.loss = instantiate(loss)

        if metrics is not None:
            self.metrics = {
                key: instantiate(value)
                for key, value in metrics.items()
            }
        else:
            self.metrics = {}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)

        for metric_name, metric in self.metrics.items():
            self.log(f"train_{metric_name}", metric(y_hat, y))

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)

        for metric_name, metric in self.metrics.items():
            self.log(f"val_{metric_name}", metric(y_hat, y), prog_bar=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)

        metrics = {}
        
        loss = self.loss(y_hat, y)
        self.log("test_loss", loss)
        metrics["test_loss"] = loss

        for metric_name, metric_fn in self.metrics.items():
            self.log(f"test_{metric_name}", metric := metric_fn(y_hat, y))
            metrics[f"test_{metric_name}"] = metric

        return metrics
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch 
        
        y_hat = self(x) 
    
        if y_hat.shape[1] == 1:
            return (torch.sigmoid(y_hat) > 0.5).int()
        else:
            return torch.argmax(y_hat, dim=1)

    def configure_optimizers(self):
        optimizer = instantiate(self.optimizer_conf, params=self.parameters()) if self.optimizer_conf is not None else None
        scheduler = instantiate(self.scheduler_conf, optimizer=optimizer) if self.scheduler_conf is not None else None

        return [optimizer], [scheduler]