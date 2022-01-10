from argparse import ArgumentParser

import numpy as np
import torch
from loguru import logger
from pl_bolts.models.vision.unet import UNet
from pytorch_lightning import LightningModule
from skimage.util.shape import view_as_blocks
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from oxeo.water.models import Predictor
from oxeo.water.models.utils import resize_sample


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
        self.criterion = CrossEntropyLoss()
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
        x = self.preprocess(x)
        return self.net(x)

    def shared_step(self, batch, batch_nb):
        img = batch["image"].float()
        label = batch["label"]  # (B, 1, H, W)

        pred = self(img)
        loss = self.criterion(pred, label)
        return loss

    def training_step(self, batch, batch_nb):
        loss = self.shared_step(self, batch)

        self.log("train/loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(self, batch)

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
        batch_size=16,
        ckpt_path: str = "gs://oxeo-models/last.ckpt",
        input_channels: int = 6,
        num_classes: int = 3,
        chip_size: int = 250,
        fs=None,
    ):
        self.model = Segmentation2D.load_from_checkpoint(
            fs.open(ckpt_path), input_channels=input_channels, num_classes=num_classes
        )
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.chip_size = chip_size
        self.model.eval()

    def predict(self, sample, target_size=None):
        original_shape = sample["image"].shape

        sample = resize_sample(sample, target_size)

        input = sample["image"].numpy()
        revisits = input.shape[0]
        bands = input.shape[1]
        H = input.shape[2]
        W = input.shape[3]

        arr = (
            view_as_blocks(input, (revisits, bands, self.chip_size, self.chip_size))
            .reshape(-1, revisits, bands, self.chip_size, self.chip_size)
            .astype(np.int16)
        )
        block_shape = arr.shape
        arr = np.vstack(arr)  # stack all revisits
        preds = []
        logger.info(
            f"Starting prediction using batch_size of {self.batch_size} for {revisits} revisits."
        )
        for patch in tqdm(range(0, arr.shape[0], self.batch_size)):
            input_tensor = torch.as_tensor(arr[patch : patch + self.batch_size]).float()

            current_pred = self.model(input_tensor)
            current_pred = torch.softmax(current_pred, dim=1)
            current_pred = torch.argmax(current_pred, 1)

            preds.extend(current_pred.data.numpy())

        preds = np.array(preds)

        preds = preds.reshape(
            (block_shape[0], block_shape[1], self.chip_size, self.chip_size)
        )

        preds = preds.reshape(
            (
                block_shape[0],
                block_shape[1],
                self.chip_size,
                self.chip_size,
            ),
            order="F",
        )
        preds = reconstruct_from_patches(preds, revisits, self.chip_size, H, W)
        preds = resize_sample(torch.as_tensor(preds), original_shape[-1])
        return preds.numpy()


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
