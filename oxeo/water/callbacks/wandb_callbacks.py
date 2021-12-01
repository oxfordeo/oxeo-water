import torch
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LoggerCollection, WandbLogger

import wandb


def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    """Safely get Weights&Biases logger from Trainer."""

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    raise Exception(
        "You are using wandb related callback, but WandbLogger was not found for some reason...",
    )


class LogImagePredictions(Callback):
    """Logs a validation batch and their predictions to wandb.
    Example adapted from:
        https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY
    """

    def __init__(self, num_samples: int = 8):
        super().__init__()
        self.num_samples = num_samples
        self.ready = True

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.ready:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment

            # get a validation batch from the validation dat loader
            val_samples = next(iter(trainer.datamodule.val_dataloader()))
            val_imgs = val_samples["data"].float()
            val_labels = val_samples["label"].unsqueeze(1)

            # run the batch through the network
            val_imgs = val_imgs.to(device=pl_module.device)

            logits = pl_module(val_imgs)

            preds = torch.sigmoid(logits)

            # log the images as wandb Image
            experiment.log(
                {
                    "inputs": [
                        wandb.Image(x, caption="input")
                        for x in val_imgs[: self.num_samples, [3, 2, 1], :, :]
                    ],
                    "y_preds": [
                        wandb.Image(x.float(), caption="y_pred")
                        for x in preds[: self.num_samples]
                    ],
                    "y_true": [
                        wandb.Image(x.float(), caption="y_true")
                        for x in val_labels[: self.num_samples]
                    ],
                },
            )
