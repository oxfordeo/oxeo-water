import os
from argparse import ArgumentParser

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from satextractor.models import constellation_info
from satools.io import ConstellationData
from torchvision.transforms import Compose

from oxeo.water.callbacks.wandb_callbacks import LogImagePredictions
from oxeo.water.datamodules import ConstellationDataModule
from oxeo.water.datamodules import transforms as oxtransforms
from oxeo.water.models.segmentation import Segmentation2D

if __name__ == "__main__":

    pl.seed_everything(1337)

    # Args ###################################################
    parser = ArgumentParser()
    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)
    # model args
    parser = Segmentation2D.add_model_specific_args(parser)

    parser = ConstellationDataModule.add_argparse_args(parser)

    parser.add_argument("--run", default=None, type=str)
    parser.add_argument("--project", default="oxeo", type=str)
    parser.add_argument("--visible_gpus", default="0", type=str)
    parser.add_argument("--checkpoint_dir", default="checkpoints", type=str)
    parser.add_argument("--input_channels", default=3, type=int)
    parser.add_argument("--premodel_ckpt", default=None, type=str)
    parser.add_argument("--train_index_map", default=None, type=str)
    parser.add_argument("--val_index_map", default=None, type=str)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus
    # model
    model = Segmentation2D(num_classes=1, **args.__dict__)

    print(args)

    # Data ###################################################

    # Define some transforms
    transform = None

    paths = [
        "oxeo-water/prod/43_P_10000_63_131",
        "oxeo-water/prod/43_P_10000_63_132",
        "oxeo-water/prod/43_P_10000_64_131",
        "oxeo-water/prod/43_P_10000_64_132",
    ]

    constellations = ["sentinel-2"]
    all_paths = {kk: [f"gs://{path}" for path in paths] for kk in constellations}

    data_sen2 = ConstellationData(
        "sentinel-2",
        bands=list(constellation_info.SENTINEL2_BAND_INFO.keys()),
        paths=all_paths["sentinel-2"],
        height=1000,
        width=1000,
    )

    data_labels = ConstellationData(
        "sentinel-2",
        bands=["pekel"],
        paths=all_paths["sentinel-2"],
        height=1000,
        width=1000,
    )

    train_index_map = pd.read_csv(args.train_index_map, header=None).values
    val_index_map = pd.read_csv(args.val_index_map, header=None).values

    patch_size = 100

    train_constellation_regions = {"data": [[data_sen2]], "label": [[data_labels]]}
    val_constellation_regions = {"data": [[data_sen2]], "label": [[data_labels]]}

    dm = ConstellationDataModule(
        train_constellation_regions=train_constellation_regions,
        val_constellation_regions=val_constellation_regions,
        patch_size=patch_size,
        train_index_map=train_index_map,
        val_index_map=val_index_map,
        preprocess=Compose(
            [
                oxtransforms.SelectConstellation("sentinel-2"),
                oxtransforms.SelectBands(["B04", "B03", "B02"]),
                oxtransforms.Compute(),
            ]
        ),
        transforms=None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Train ##################################################

    run = args.run

    wandb_logger = WandbLogger(name=run, project=args.project, entity="oxeo")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=args.checkpoint_dir,
        filename=run + "-{epoch:02d}-{val_loss:.3f}",
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

    trainer.fit(model, datamodule=dm)
