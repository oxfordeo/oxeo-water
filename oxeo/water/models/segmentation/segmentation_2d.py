from argparse import ArgumentParser

import torch
from pl_bolts.models.vision.unet import UNet
from pytorch_lightning import LightningModule
from torch.nn import BCEWithLogitsLoss


class Segmentation2D(LightningModule):
    def __init__(
        self,
        lr: float = 0.01,
        num_classes: int = 19,
        input_channels: int = 3,
        num_layers: int = 5,
        features_start: int = 64,
        bilinear: bool = False,
        **kwargs,
    ):
        """
        Basic model for semantic segmentation. Uses UNet architecture by default.
        The default parameters in this model are for the KITTI dataset. Note, if you'd like to use this model as is,
        you will first need to download the KITTI dataset yourself. You can download the dataset `here.
        <http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015>`_
        Implemented by:
            - `Annika Brundyn <https://github.com/annikabrundyn>`_
        Args:
            num_layers: number of layers in each side of U-net (default 5)
            features_start: number of features in first layer (default 64)
            bilinear: whether to use bilinear interpolation (True) or transposed convolutions (default) for upsampling.
            lr: learning (default 0.01)
        """
        super().__init__()

        self.num_classes = num_classes
        self.input_channels = input_channels
        self.num_layers = num_layers
        self.features_start = features_start
        self.bilinear = bilinear
        self.lr = lr

        self.criterion = BCEWithLogitsLoss()
        self.net = UNet(
            num_classes=num_classes,
            input_channels=input_channels,
            num_layers=self.num_layers,
            features_start=self.features_start,
            bilinear=self.bilinear,
        )

    def forward(self, x):
        if len(x.shape) == 5:  # (B, C, T, H, W)
            x = torch.median(x, 2)[0]

        return self.net(x)

    def training_step(self, batch, batch_nb):
        img = batch["data"].float()
        label = batch["label"].unsqueeze(1)  # (B, 1, H, W)

        pred = self(img)
        loss = self.criterion(pred, label.float())

        torch.sigmoid(pred)
        self.log("train/loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        img = batch["data"].float()
        label = batch["label"].unsqueeze(1)

        pred = self(img)

        loss = self.criterion(pred, label.float())

        self.log("val/loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        # sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return opt  # [opt], [sch]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--lr",
            type=float,
            default=0.01,
            help="adam: learning rate",
        )
        parser.add_argument(
            "--num_layers",
            type=int,
            default=5,
            help="number of layers on u-net",
        )
        parser.add_argument(
            "--features_start",
            type=float,
            default=64,
            help="number of features in first layer",
        )
        parser.add_argument(
            "--bilinear",
            action="store_true",
            default=False,
            help="whether to use bilinear interpolation or transposed",
        )

        return parser
