from typing import (
    cast
)

import os

import hydra
from hydra.utils import (
    instantiate
)

from omegaconf import (
    DictConfig, 
    OmegaConf
)

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

@hydra.main(config_path="conf", config_name="main", version_base=None)
def main(conf: DictConfig):
    print(OmegaConf.to_yaml(conf))

    data_module = cast(pl.LightningDataModule, instantiate(conf.data.datamodule, _recursive_=False))
    model_module = instantiate(conf.model, _recursive_=False)

    csv_logger = CSVLogger(save_dir=conf.output_dir)
    trainer = cast(pl.Trainer, instantiate(conf.trainer, logger=csv_logger))

    trainer.fit(model_module, data_module)

    checkpoint_callbacks = []

    for checkpoint_callback in trainer.checkpoint_callbacks:
        if isinstance(checkpoint_callback, ModelCheckpoint):
            checkpoint_callbacks.append(checkpoint_callback)

    for checkpoint_callback in checkpoint_callbacks:
        trainer.test(datamodule=data_module, ckpt_path=checkpoint_callback.best_model_path, weights_only=False)

    if data_module.test_dataloader is not None:
        for checkpoint_callback in checkpoint_callbacks:
            predictions = trainer.predict(datamodule=data_module, ckpt_path=checkpoint_callback.best_model_path, weights_only=False)
            predictions = torch.concat(predictions, dim=0)
            os.makedirs(os.path.join(conf.output_dir, "predictions"), exist_ok=True)
            torch.save(predictions, os.path.join(conf.output_dir, "predictions", f"{conf.output_id}.pt"))


if __name__ == "__main__":
    main()