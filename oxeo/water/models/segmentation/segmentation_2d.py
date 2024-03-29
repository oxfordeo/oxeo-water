from argparse import ArgumentParser
from functools import lru_cache
from typing import Tuple, Union

import dask
import dask.array as da
import numpy as np
import segmentation_models_pytorch as smp
import toolz
import torch
from pl_bolts.models.vision.unet import UNet
from pytorch_lightning import LightningModule
from sentinelhub import BBox, DataCollection, SentinelHubCatalog
from skimage.util.shape import view_as_blocks
from torch.nn import CrossEntropyLoss
from torchvision.transforms import Compose
from tqdm import tqdm

from oxeo.core.constants import BAND_PREDICTOR_ORDER, RESOLUTION_INFO
from oxeo.core.data import (
    SearchParams,
    get_aoi_from_stac_catalog,
    load_aoi_from_stac_as_dict,
)
from oxeo.core.logging import logger
from oxeo.core.models.tile import (
    TilePath,
    load_tile_as_dict,
    load_tile_from_stac_as_dict,
)
from oxeo.core.utils import identity
from oxeo.water.datamodules.constants import (
    CONSTELLATION_BAND_MEAN,
    CONSTELLATION_BAND_STD,
)
from oxeo.water.datamodules.transforms import ConstellationNormalize
from oxeo.water.models.base import Predictor
from oxeo.water.models.tile_utils import pad_sample, resize_sample

# model = smp.Unet(
#     encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     classes=3,                      # model output channels (number of classes in your dataset)
# )


class DeepLabSegmentation(LightningModule):
    def __init__(
        self,
        lr: float = 0.01,
        num_classes: int = 3,
        input_channels: int = 3,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.lr = lr
        self.criterion = CrossEntropyLoss()
        self.net = smp.DeepLabV3Plus(
            in_channels=self.input_channes, classes=self.num_classes
        )


class Segmentation2D(LightningModule):
    def __init__(
        self,
        lr: float = 0.01,
        num_classes: int = 19,
        input_channels: int = 3,
        model_name: str = "unet",
        # num_layers: int = 5,
        # features_start: int = 64,
        # bilinear: bool = False,
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
        self.lr = lr
        self.criterion = CrossEntropyLoss()
        self.model_name = model_name
        if self.model_name == "unet":

            self.net = UNet(
                num_classes=num_classes, input_channels=input_channels, **kwargs
            )
        elif self.model_name == "deeplab":
            self.net = smp.DeepLabV3Plus(
                in_channels=self.input_channels, classes=self.num_classes, **kwargs
            )

    def forward(self, x):
        if len(x.shape) == 5:  # (B, C, T, H, W)
            x = torch.median(x, 2)[0]
        return self.net(x)

    def shared_step(self, batch):
        img = batch["image"].float()
        label = batch["label"]  # (B, 1, H, W)

        pred = self(img)
        loss = self.criterion(pred, label)
        return loss

    def training_step(self, batch, batch_nb):
        loss = self.shared_step(batch)

        self.log("train/loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)

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

        return parser


class DaskSegmentationPredictor(Predictor):
    def __init__(
        self,
        batch_size=1,
        ckpt_path: str = "gs://oxeo-models/last.ckpt",
        input_channels: int = 6,
        num_classes: int = 3,
        chip_size: int = 250,
        fs=None,
        target_resolution: int = 10,
        model_name: str = "unet",
        **kwargs,
    ):
        self.ckpt_path = ckpt_path
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.chip_size = chip_size
        self.use_cuda = torch.cuda.is_available()

        self.target_resolution = target_resolution
        self.fs = fs
        self.model_name = model_name
        self.get_model = self.load_model

    def load_model(self):
        logger.info(f"Loading model from path {self.ckpt_path}")
        if self.fs is None:
            ckpt = self.ckpt_path
        else:
            ckpt = self.fs.open(self.ckpt_path)
        model = Segmentation2D.load_from_checkpoint(
            ckpt,
            input_channels=self.input_channels,
            num_classes=self.num_classes,
            model_name=self.model_name,
        )
        if self.use_cuda:
            model.eval().cuda()
        else:
            model.eval()
        return model

    def pad_xarray_to(self, arr, pad_to):
        pad_top = abs(arr.shape[-2] % -pad_to)
        pad_right = abs(arr.shape[-1] % -pad_to)

        return arr.pad(x=(0, pad_right), y=(0, pad_top))

    def create_patches_from_xarray(self, arr, patch_size):
        patches = []
        for t in range(0, arr.shape[0]):
            for i in range(0, arr.shape[-2], patch_size):
                for j in range(0, arr.shape[-1], patch_size):
                    patches.append(arr[t, :, i : i + patch_size, j : j + patch_size])
        return patches

    def predict_stac_aoi(
        self,
        constellation: str,
        catalog: Union[str, SentinelHubCatalog],
        data_collection: Union[str, DataCollection],
        bbox: BBox,
        time_interval: Tuple[str, str],
        search_params: SearchParams,
        resolution: int = 10,
    ):
        constellation_bands = BAND_PREDICTOR_ORDER[constellation]

        aoi = get_aoi_from_stac_catalog(
            catalog=catalog,
            data_collection=data_collection,
            bbox=bbox,
            time_interval=time_interval,
            search_params=search_params,
            resolution=resolution,
        )

        band_names = aoi.band.values
        bands_index = [list(band_names).index(name) for name in constellation_bands]
        padded_aoi = self.pad_xarray_to(aoi.isel(band=bands_index), self.chip_size)

        patches = self.create_patches_from_xarray(
            padded_aoi,
            patch_size=self.chip_size,
        )

        batches = [
            self.to_tensor(da.stack(list(batch)))
            for batch in toolz.partition_all(self.batch_size, patches)
        ]

        model = self.get_model()
        dmodel = dask.delayed(model)

        predictions = [
            self.dask_predict(dmodel, batch, constellation) for batch in batches
        ]
        return predictions, padded_aoi

    @dask.delayed
    def to_tensor(self, batch):
        return torch.as_tensor(batch)

    @dask.delayed
    def dask_predict(self, model, batch, constellation):
        constellation_bands = BAND_PREDICTOR_ORDER[constellation]
        transforms = Compose(
            [
                ConstellationNormalize(
                    CONSTELLATION_BAND_MEAN, CONSTELLATION_BAND_STD, constellation_bands
                ),
            ]
        )
        item = {}
        input_tensor = torch.as_tensor(batch, dtype=torch.int16)
        tensors = []
        for t in input_tensor:
            item["image"] = t
            item["constellation"] = constellation
            item = transforms(item)
            tensors.append(item["image"])
        tensors = torch.stack(tensors)
        if self.use_cuda:
            tensors = tensors.cuda()
        current_pred = model(tensors)
        del tensors
        current_pred = torch.softmax(current_pred, dim=1)
        current_pred = torch.argmax(current_pred, 1)
        current_pred = current_pred.data.type(torch.uint8)
        if self.use_cuda:
            current_pred = current_pred.cpu()
        current_pred = current_pred.numpy()
        return current_pred


class Segmentation2DPredictor(Predictor):
    def __init__(
        self,
        batch_size=16,
        ckpt_path: str = "gs://oxeo-models/last.ckpt",
        input_channels: int = 6,
        num_classes: int = 3,
        chip_size: int = 250,
        fs=None,
        target_resolution: int = 10,
        model_name: str = "unet",
        **kwargs,
    ):
        self.ckpt_path = ckpt_path
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.chip_size = chip_size
        self.use_cuda = torch.cuda.is_available()

        self.target_resolution = target_resolution
        self.fs = fs
        self.model_name = model_name
        self.get_model = self.lazy_load_model()

    def lazy_load_model(self):
        @lru_cache(maxsize=None)
        def load_model():
            logger.info(f"Loading model from path {self.ckpt_path}")
            if self.fs is None:
                ckpt = self.ckpt_path
            else:
                self.fs.open(self.ckpt_path)
            model = Segmentation2D.load_from_checkpoint(
                ckpt,
                input_channels=self.input_channels,
                num_classes=self.num_classes,
                model_name=self.model_name,
            )
            if self.use_cuda:
                model.eval().cuda()
            else:
                model.eval()
            return model

        return load_model

    def predict_stac_aoi(
        self,
        constellation: str,
        catalog: Union[str, SentinelHubCatalog],
        data_collection: Union[str, DataCollection],
        bbox: BBox,
        time_interval: Tuple[str, str],
        search_params: SearchParams,
        revisit: slice,
        median: bool = False,
    ):
        constellation_bands = BAND_PREDICTOR_ORDER[constellation]
        sample = load_aoi_from_stac_as_dict(
            catalog,
            data_collection,
            bbox,
            time_interval,
            search_params,
            constellation_bands,
            revisit,
            median,
        )
        # convert sample img to tensor
        sample["image"] = torch.from_numpy(sample["image"])
        resampled_shape = sample["image"].shape
        sample = pad_sample(sample, pad_to=self.chip_size)

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
        logger.debug(
            f"Starting prediction using batch_size of {self.batch_size} for {revisits} revisits."
        )

        item = {}
        model = self.get_model()

        transforms = Compose(
            [
                ConstellationNormalize(
                    CONSTELLATION_BAND_MEAN, CONSTELLATION_BAND_STD, bands
                ),
            ]
        )

        for i, patch in enumerate(tqdm(range(0, arr.shape[0], self.batch_size))):
            logger.debug(f"Pred loop {i} of {arr.shape[0]} with {self.batch_size=}")
            tensors = []
            input_tensor = torch.as_tensor(
                arr[patch : patch + self.batch_size], dtype=torch.int16
            )
            for t in input_tensor:
                item["image"] = t
                item["constellation"] = constellation
                item = transforms(item)
                tensors.append(item["image"])
            tensors = torch.stack(tensors)
            if self.use_cuda:
                tensors = tensors.cuda()
            current_pred = model(tensors)
            del tensors
            current_pred = torch.softmax(current_pred, dim=1)
            current_pred = torch.argmax(current_pred, 1)
            current_pred = current_pred.data.type(torch.uint8)
            if self.use_cuda:
                current_pred = current_pred.cpu()
            current_pred = current_pred.numpy()
            preds.extend(current_pred)
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
        # remove the pad used for predictions:
        preds = preds[..., : resampled_shape[-2], : resampled_shape[-1]]
        # resize to original size
        preds = resize_sample(
            torch.as_tensor(preds),
            sample_resolution=self.target_resolution,
            target_resolution=RESOLUTION_INFO[constellation],
        )
        return preds.numpy()

    def predict(
        self,
        tile_path: TilePath = None,
        revisit=None,
        fs=None,
        use_stac=False,
        stac_kwargs=None,
    ):
        if fs is not None:
            fs_mapper = fs.get_mapper
        else:
            fs_mapper = identity

        if use_stac:
            sample = load_tile_from_stac_as_dict(
                catalog_url=stac_kwargs["catalog_url"],
                collections=stac_kwargs["collections"],
                tile=tile_path.tile,
                revisit=revisit,
                bands=self.bands,
                chunk_aligned=False,
                resolution=None,
            )
        else:
            sample = load_tile_as_dict(
                fs_mapper=fs_mapper,
                tile_path=tile_path,
                masks=(),
                revisit=revisit,
                bands=self.bands,
            )
        sample = resize_sample(
            sample,
            sample_resolution=RESOLUTION_INFO[tile_path.constellation],
            target_resolution=self.target_resolution,
        )
        constellation_bands = BAND_PREDICTOR_ORDER[tile_path.constellation]
        transforms = Compose(
            [
                ConstellationNormalize(
                    CONSTELLATION_BAND_MEAN, CONSTELLATION_BAND_STD, constellation_bands
                ),
            ]
        )
        resampled_shape = sample["image"].shape
        sample = pad_sample(sample, pad_to=self.chip_size)

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
        logger.debug(
            f"Starting prediction using batch_size of {self.batch_size} for {revisits} revisits."
        )

        item = {}
        model = self.get_model()
        for i, patch in enumerate(tqdm(range(0, arr.shape[0], self.batch_size))):
            logger.debug(f"Pred loop {i} of {arr.shape[0]} with {self.batch_size=}")
            tensors = []
            input_tensor = torch.as_tensor(
                arr[patch : patch + self.batch_size], dtype=torch.int16
            )
            for t in input_tensor:
                item["image"] = t
                item["constellation"] = tile_path.constellation
                item = transforms(item)
                tensors.append(item["image"])
            tensors = torch.stack(tensors)
            if self.use_cuda:
                tensors = tensors.cuda()
            current_pred = model(tensors)
            del tensors
            current_pred = torch.softmax(current_pred, dim=1)
            current_pred = torch.argmax(current_pred, 1)
            current_pred = current_pred.data.type(torch.uint8)
            if self.use_cuda:
                current_pred = current_pred.cpu()
            current_pred = current_pred.numpy()
            preds.extend(current_pred)
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
        # remove the pad used for predictions:
        preds = preds[..., : resampled_shape[-2], : resampled_shape[-1]]
        # resize to original size
        preds = resize_sample(
            torch.as_tensor(preds),
            sample_resolution=self.target_resolution,
            target_resolution=RESOLUTION_INFO[tile_path.constellation],
        )
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


@dask.delayed
def stack_preds(preds):
    return da.vstack(preds)


@dask.delayed
def reconstruct_image_from_patches(stack, revisits, target_h, target_w, patch_size):
    num_patches_h = target_h // patch_size
    num_patches_w = target_w // patch_size
    return (
        stack.reshape(revisits, num_patches_h, num_patches_w, patch_size, patch_size)
        .swapaxes(2, 3)
        .reshape(revisits, target_h, target_w)
    )


@dask.delayed
def reduce_to_timeseries(stack):
    return (stack == 1).sum(axis=(1, 2))
