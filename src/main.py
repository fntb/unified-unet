from typing import cast
import os
import random

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger


@hydra.main(config_path="conf", config_name="main", version_base=None)
def main(conf: DictConfig):

    # Reproducibility (global)
    seed = int(conf.get("seed", 0))
    pl.seed_everything(seed, workers=True)
    random.seed(seed)

    # Determinism flags (CUDA)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Optional: stricter determinism (can raise errors depending on ops/backends)
    # torch.use_deterministic_algorithms(True)

    # Optional: matmul precision (not required for determinism)
    # torch.set_float32_matmul_precision("high")

    print(OmegaConf.to_yaml(conf))

    # Build objects from Hydra
    data_module = cast(pl.LightningDataModule, instantiate(conf.data.datamodule, _recursive_=False))
    model_module = instantiate(conf.model, _recursive_=False)

    csv_logger = CSVLogger(save_dir=conf.output_dir)
    trainer = cast(pl.Trainer, instantiate(conf.trainer, logger=csv_logger))

    # Train
    trainer.fit(model_module, data_module)

    # Test with best checkpoint(s)
    checkpoint_callbacks = [
        cb for cb in trainer.checkpoint_callbacks
        if isinstance(cb, ModelCheckpoint)
    ]

    for cb in checkpoint_callbacks:
        trainer.test(datamodule=data_module, ckpt_path=cb.best_model_path, weights_only=False)

    # Save predictions (optional)
    if data_module.test_dataloader is not None:
        for cb in checkpoint_callbacks:
            preds = trainer.predict(datamodule=data_module, ckpt_path=cb.best_model_path, weights_only=False)
            preds = torch.concat(preds, dim=0)

            os.makedirs(os.path.join(conf.output_dir, "predictions"), exist_ok=True)
            torch.save(preds, os.path.join(conf.output_dir, "predictions", f"{conf.output_id}.pt"))


if __name__ == "__main__":
    main()