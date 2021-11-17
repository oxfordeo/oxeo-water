import glob
import os
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from oxeo.water.callbacks.wandb_callbacks import LogImagePredictions
from oxeo.water.datamodules import SegmentationDataModule
from oxeo.water.models.segmentation import Segmentation2D

if __name__ == "__main__":

    pl.seed_everything(1337)

    # Args ###################################################
    parser = ArgumentParser()
    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)
    # model args
    parser = Segmentation2D.add_model_specific_args(parser)

    parser = SegmentationDataModule.add_argparse_args(parser)

    parser.add_argument("--run", default=None, type=str)
    parser.add_argument("--project", default="oxeo", type=str)
    parser.add_argument("--visible_gpus", default="0", type=str)
    parser.add_argument("--checkpoint_dir", default="checkpoints", type=str)
    parser.add_argument("--input_channels", default=3, type=int)
    parser.add_argument("--premodel_ckpt", default=None, type=str)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus
    # model
    model = Segmentation2D(num_classes=1, **args.__dict__)

    print(args)

    # Data ###################################################

    # Define some transforms
    transform = None

    train_dams = ["harangi", "hemavathy", "krishnaraja_sagar"]
    val_dams = ["kabini"]
    train_paths = []
    for d in train_dams:
        train_paths.extend(
            glob.glob(f"/home/fran/repos/oxeo-water/data/oxeo-water/eo/{d}/*/*")
        )
    val_paths = glob.glob("/home/fran/repos/oxeo-water/data/oxeo-water/eo/kabini/*/*")

    # Instantiate it (we are using here the Embeddings data module, without labels)
    sdm = SegmentationDataModule(
        transforms=transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_paths=train_paths,
        val_paths=val_paths,
    )

    # Train ##################################################

    run = args.run

    wandb_logger = WandbLogger(name=run, project=args.project, entity="oxeo")

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="{epoch}_{val_loss}",
    )

    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=wandb_logger,
        callbacks=[
            checkpoint_callback,
            LogImagePredictions(),
            # LearningRateMonitor(logging_interval="step"),
        ],
        precision=args.precision,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        max_epochs=args.max_epochs,
    )

    trainer.fit(model, datamodule=sdm)
