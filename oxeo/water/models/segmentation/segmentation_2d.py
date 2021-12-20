from argparse import ArgumentParser

import numpy as np
import torch
from loguru import logger
from pl_bolts.models.vision.unet import UNet
from pytorch_lightning import LightningModule
from skimage.util.shape import view_as_blocks
from torch.nn import BCEWithLogitsLoss

from oxeo.water.models import Predictor


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
        self.save_hyperparameters(logger=False)

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
        img = batch["image"].float()
        label = batch["pekel"]  # (B, 1, H, W)

        pred = self(img)
        loss = self.criterion(pred, label.float())

        self.log("train/loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        img = batch["image"].float()
        label = batch["pekel"]

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


class Segmentation2DPredictor(Predictor):
    def __init__(
        self,
        batch_size=128,
        ckpt_path: str = None,
        input_channels: int = 13,
        num_classes: int = 1,
    ):
        self.model = Segmentation2D.load_from_checkpoint(
            ckpt_path, input_channels=input_channels, num_classes=num_classes
        )
        self.batch_size = batch_size
        self.model.eval().cuda()

    def predict(self, input):
        logger.info("Moving model to GPU.")
        self.model.cuda()
        input_view = (
            view_as_blocks(input, (input.shape[0], input.shape[1], 100, 100))
            .reshape(-1, input.shape[0], input.shape[1], 100, 100)
            .astype(np.int16)
        )
        preds = []
        for revisit in range(input_view.shape[1]):
            for patch in range(0, input_view.shape[0], self.batch_size):
                input_tensor = torch.as_tensor(
                    input_view[patch : patch + self.batch_size, revisit, :, :]
                ).float()
                preds.extend(
                    torch.sigmoid(self.model(input_tensor.cuda())).data.cpu().numpy()
                )

        logger.info("Moving model to CPU.")
        self.model.cpu()
        preds = np.array(preds)
        preds = preds.reshape(
            (input_view.shape[0], input_view.shape[1], 100, 100), order="F"
        )
        return reconstruct_from_patches(
            preds, input_view.shape[1], 100, input.shape[-2], input.shape[-1]
        )


def reconstruct_from_patches(
    images, revisits, patch_size, target_size_rows, target_size_cols
):
    block_n = 0
    h_stack = []
    rec_img = []
    for revisit in range(revisits):
        v_stack = []
        for _ in range(target_size_rows // patch_size):
            h_stack = []
            for _ in range(target_size_cols // patch_size):
                h_stack.append(images[block_n, revisit, :, :])
                block_n += 1

            v_stack.append(np.hstack(h_stack))

        rec_img.append(np.vstack(v_stack))
        block_n = 0
    return np.array(rec_img)
