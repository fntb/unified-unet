
import hydra
from hydra.utils import (
    instantiate
)

from omegaconf import (
    DictConfig, 
    OmegaConf
)

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

@hydra.main(config_path="conf", config_name="main", version_base=None)
def main(conf: DictConfig):
    print(OmegaConf.to_yaml(conf))

    data_module = instantiate(conf.data, _recursive_=False)
    model_module = instantiate(conf.model, _recursive_=False)

    csv_logger = CSVLogger(save_dir=conf.output_dir)
    trainer = instantiate(conf.trainer, logger=csv_logger)

    trainer.fit(model_module, data_module)

    checkpoint_callback = None

    for callback in trainer.callbacks:
        if isinstance(callback, ModelCheckpoint):
            checkpoint_callback = callback
            break

    if checkpoint_callback is not None:
        trainer.test(datamodule=data_module, ckpt_path=checkpoint_callback.best_model_path, weights_only=False)


if __name__ == "__main__":
    main()