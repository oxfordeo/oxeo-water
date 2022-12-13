<img src="oxeo_logo.png" alt="oxeo logo" width="600"/>

[OxEO](https://www.oxfordeo.com/) is an earth observation water risk company. This repository builds and deploys OxEO's data pipeline Flows via Prefect. OxEO's data service is comprised of three repos: [oxeo-flows](https://github.com/oxfordeo/oxeo-flows), [oxeo-water](https://github.com/oxfordeo/oxeo-water), and [oxeo-api](https://github.com/oxfordeo/oxeo-api). This work was generously supported by the [European Space Agency Φ-lab](https://philab.esa.int/) and [World Food Programme (WFP) Innovation Accelerator](https://innovation.wfp.org/) as part of the [EO & AI for SDGs Innovation Initiative](https://wfpinnovation.medium.com/how-can-earth-observation-and-artificial-intelligence-help-people-in-need-5e56efc5c061).

Copyright © 2022 Oxford Earth Observation Ltd.

---

# oxeo-water
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A research repository for water inference in multi-spectral earth observation imagery: Landsat-5, Landsat-7, Landsat-8, and Sentinel-2.



## Installation

This repo is shipped with several installation options: `dev`, `ml`, `sql`, and `graph`.

In general, any option can be installed using `pip`:

```
pip install .[<option>]
```

### Machine Learning (ml)

The `ml` option installs all the prerequisites for machine learning training and inference.
Machine learning training uses [hydra](https://hydra.cc/docs/intro/) for configuration management, [pytorch](https://pytorch.org/) and [pytorch-lightning](https://www.pytorchlightning.ai/) for training, [weights-and-biases](https://wandb.ai/site) for logging and MLOps, and Google Cloud Storage for storing checkpoints.

### Database options (sql & graph)

The `sql` option enables training and inference to communicate with a PostgreSQL database. The `graph` option is experimental and is for building a graph database instead of a conventional relational database.

### Development

The `dev` option install pre-commit, diagnostics, and testing packages.
Use it when setting up this repo for development and contributions.

```
pip install -e .[dev]
pre-commit install
```

Run tests:
```
tox
```

## Useage: Training

The model training is fully configurable with hydra. The configs are divided in modules in oxeo/water/configs.
There you can define configs for datamodule, trainer, etc.
In most cases you won't want to touch those configs but just create a new `experiment` config inside `oxeo/water/configs/experiment`.

There you can find the configs used to train the existing cnn (`unet_semseg_all_tiles.yaml`).

The only thing you will want to change is the tiles used for training and validation :
```yaml
datamodule:
  train_constellation_tile_ids:
    sentinel-2: [18_L_10000_24_895],
    landsat-5 : [18_L_10000_24]
```

Other important parameters are those managing the cache as the training is performed on the fly reading from the bucket and storing partial loads in local disk.

```yaml
  chip_size: 256
  revisits_per_epoch: 1024 # this is how many random revisits are going to be cached per epoch
  samples_per_revisit: 256 # this is how many samples will be taken from each cached revisit
  batch_size: 64
  num_workers: 6
  pin_memory: False
  cache_dir: "../../cache" # the directory where the cache will be stored. The default is logs/experiment/runs/cache
  cache_bytes: 1e11 # The space in disk you want to use for the cache. A bit less than a 1TB
```

Once you configure it all, you just have to run the following script:
`python oxeo/water/run.py experiment=unet_semseg_all_tiles_ft.yaml`

And that's all! It will store the results on `Wandb` and automatically train from bucket tiles.

### Fine-tuning
If you want to continue training an existing model you just have to define the `resume_from_checkpoint` parameter.
An example can be found in `unet_semseg_all_tiles_ft.yaml`

## Useage: Prediction
### Local
If you need to run a local prediction you can use the notebook `Unet Predictor` as a starting example.
### Prefect
After the training is complete you will need to upload the best checkpoint to the `oxeo/models/` bucket.
Once it is there you can run prefect passing that chackpoint.

A common prefect config to run the cnn is:
```json
 {
  "run_name": "la-alumbrera", # important to find it later in the bigquery table
  "bands": ["nir", "red", "green", "blue", "swir1", "swir2"],
  "bucket": "oxeo-water",
  "cktp_path": "gs://oxeo-models/semseg/epoch_012.ckpt",
  "cnn_batch_size": 4, # keep it small to avoid mem errors
  "constellations": ["sentinel-2", "landsat-5", "landsat-7", "landsat-8"],
  "cpu_per_worker": 8,
  "credentials": "token.json",
  "gpu_per_worker": 0,
  "memory_per_worker": "32G",
  "model_name": "cnn",
  "n_workers": 16,
  "project": "oxeo-main",
  "revisit_chunk_size": 2, # keep it low to avoid mem errors
  "root_dir": "prod",
  "target_size": 1000,
  "timeseries_label": 1, # if set to 1 it use water mask, if 2 it uses cloud mask
  "water_list": [51318547]
}
```

## How to see changes in oxeo-water reflected in Prefect
If you change the code in this repository, it won't be automatically updated. You have to go to `actions` in `oxeo-flows` repository and
run the `build-images` action..

## Metrics
There is a notebook called `Baseline Metrics` where you can find the queries to retrieve the timeseries results (pekel and cnn) and plot them together.
