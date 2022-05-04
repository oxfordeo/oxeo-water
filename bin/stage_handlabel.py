import os

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from google.cloud import storage
from torchvision.transforms import Compose

from oxeo.core import constants
from oxeo.satools.io import ConstellationData, create_index_map
from oxeo.water.datamodules import ConstellationDataModule
from oxeo.water.datamodules import transforms as oxtransforms


@click.group()
def cli():
    pass


@cli.command()
@click.argument("target")
@click.argument("paths")
@click.argument("constellations")
@click.argument("mask")
@click.argument("start_datetime")
@click.argument("end_datetime")
@click.option("--patch-size", type=int, default=512)
def swipe_labeller(target, paths, constellations, mask, start_dt, end_dt, patch_size):

    swipe_labeller_fn(target, paths, constellations, mask, start_dt, end_dt, patch_size)


def save_file_to_bucket(target_directory: str, source_directory: str):
    """Function to save file to a bucket.
    Args:
        target_directory (str): Destination file path.
        source_directory (str): Source file path
    Returns:
        None: Returns nothing.
    Examples:
        >>> target_directory = 'target/path/to/file/.pkl'
        >>> source_directory = 'source/path/to/file/.pkl'
        >>> save_file_to_bucket(target_directory)
    """

    client = storage.Client()

    bucket_id = target_directory.split("/")[0]
    file_path = "/".join(target_directory.split("/")[1:])

    bucket = client.get_bucket(bucket_id)

    # get blob
    blob = bucket.blob(file_path)

    # upload data
    blob.upload_from_filename(source_directory)

    return None


def write_figure(arr, figure_path):

    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    axs[0].imshow((arr["data"] / 4000).clip(0, 1).transpose(1, 2, 0))
    axs[1].imshow(np.squeeze(arr["label"]).astype(float))

    fig.savefig(figure_path)
    plt.cla()
    plt.clf()
    plt.close()

    return 1


def swipe_labeller_fn(
    target,
    paths,
    constellations,
    mask,
    start_datetime,
    end_datetime,
    patch_size,
    index_map_path=None,
):

    all_paths = {kk: [f"gs://{path}" for path in paths] for kk in constellations}

    data = {}
    for kk in constellations:
        data[kk] = dict(
            data=ConstellationData(
                kk,
                bands=list(constants.SENTINEL2_BAND_INFO.keys()),
                paths=all_paths[kk],
                height=1000,
                width=1000,
            ),
            labels=ConstellationData(
                kk, bands=[mask], paths=all_paths[kk], height=1000, width=1000
            ),
        )

    train_constellation_regions = {
        "data": [[vv["data"] for kk, vv in data.items()]],
        "label": [[vv["labels"] for kk, vv in data.items()]],
    }

    if index_map_path is None:

        train_index_map = create_index_map(
            train_constellation_regions,
            date_range=(start_datetime, end_datetime),
            patch_size=patch_size,
            output=index_map_path,
        )

        train_index_map = np.array(train_index_map)

    else:
        train_index_map = pd.read_csv(index_map_path, header=None).values

    dm = ConstellationDataModule(
        train_constellation_regions=train_constellation_regions,
        val_constellation_regions=train_constellation_regions,
        patch_size=patch_size,
        train_index_map=train_index_map,
        val_index_map=train_index_map,
        preprocess=Compose(
            [  # how to into multi-constellation?
                oxtransforms.SelectConstellation("sentinel-2"),
                oxtransforms.SelectBands(["B04", "B03", "B02"]),
                oxtransforms.Compute(),
            ]
        ),
        transforms=None,
        batch_size=16,
        num_workers=4,
    )

    dm.setup()

    dl = dm.train_dataloader()

    for idx, arr in iter(enumerate(dl.dataset)):

        fig_path = os.path.join(
            os.getcwd(),
            "_".join([str(el) for el in train_index_map[idx].tolist()]) + ".png",
        )
        target_path = os.path.join(target, "samples", os.path.split(fig_path)[-1])

        write_figure(arr, fig_path)

        save_file_to_bucket(target_path, fig_path)

        os.remove(fig_path)

    return 1


if __name__ == "__main__":

    paths = [
        "oxeo-water/prod/43_P_10000_63_131",
        "oxeo-water/prod/43_P_10000_63_132",
        "oxeo-water/prod/43_P_10000_64_131",
        "oxeo-water/prod/43_P_10000_64_132",
    ]

    constellations = ["sentinel-2"]

    target = "oxeo-handlabelling/swipelabel/s2-pekel-demo"

    swipe_labeller_fn(
        target=target,
        paths=paths,
        constellations=constellations,
        mask="pekel",
        start_datetime="2015-01-01",
        end_datetime="2017-01-01",
        patch_size=512,
        index_map_path=None,
    )
