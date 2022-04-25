from collections import defaultdict
from datetime import datetime
from functools import reduce
from typing import Dict, List, Tuple

import attr
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr

from oxeo.core.logging import logger


def strdates_to_datetime(dates: List[str]) -> np.ndarray:
    return np.array([datetime.fromisoformat(x[:10]) for x in sorted(list(dates))])


def id_f(x):
    return x


@attr.s
class ConstellationData:
    name: str = attr.ib()
    bands: List[str] = attr.ib()
    paths: List[str] = attr.ib()


def constellation_dataarray(
    constellation: ConstellationData, data_path="data", fs_mapper=None
):
    # Get all timestamps and their union.
    # We want all patches in the same constellation to share the dates

    fs_mapper = id_f if fs_mapper is None else fs_mapper

    timestamp_paths = [
        f"{p}/{constellation.name}/timestamps" for p in constellation.paths
    ]

    x_coords = [p.split("_")[-2] for p in constellation.paths]
    y_coords = [p.split("_")[-1] for p in constellation.paths]

    all_timestamps = [
        strdates_to_datetime(da.from_zarr(fs_mapper(time_path)).compute())
        for time_path in timestamp_paths
    ]

    timestamps = reduce(np.intersect1d, all_timestamps)
    all_timestamps_masks = [
        np.in1d(t, timestamps, assume_unique=True) for t in all_timestamps
    ]

    arrays = defaultdict(dict)
    example = None
    for i, pp in enumerate(constellation.paths):
        x, y = pp.split("_")[-2:]
        arr_path = f"{pp}/{constellation.name}/{data_path}"
        da_arr = da.from_zarr(fs_mapper(arr_path))[all_timestamps_masks[i]]
        if len(da_arr.shape) == 3:  # it has no bands
            # add band channel (can be labels)
            da_arr = da.stack([da_arr], axis=1)
        logger.debug(f"Inserted tile at {x=}, {y=}")
        arrays[y][x] = da_arr
        if example is None:
            example = da_arr

    for y in sorted(set(y_coords), reverse=True):
        for x in sorted(set(x_coords)):
            if y not in arrays.keys() or x not in arrays[y].keys():
                logger.debug(f"Created NAN tile at {x=}, {y=}")
                arrays[y][x] = da.full(
                    example.shape,
                    fill_value=np.nan,
                    chunks=example.chunksize,
                    dtype=example.dtype,
                )

    arrays = [
        [arrays[row][col] for col in sorted(arrays[row])]
        for row in sorted(arrays, reverse=True)
    ]
    block = da.block(arrays)

    return xr.DataArray(
        block,
        coords={"revisits": timestamps, "bands": constellation.bands},
        dims=["revisits", "bands", "height", "width"],
    )


def constellations_dataset(
    constellations: List[ConstellationData], data_path="data", fs_mapper=None
):
    # init varrays
    constellation_data_arrs = []
    constellation_timestamps = []
    constellation_bands = []
    constellation_attrs = []
    for c_data in constellations:
        data_arr = constellation_dataarray(c_data, data_path, fs_mapper)
        constellation_data_arrs.append(data_arr)
        constellation_timestamps.append(data_arr.coords["revisits"].data)
        constellation_bands.append(data_arr.coords["bands"].data)
        constellation_attrs.extend(c_data.paths)

    all_timestamps = reduce(np.union1d, constellation_timestamps)
    all_bands = reduce(np.union1d, constellation_bands)
    for i, c_da in enumerate(constellation_data_arrs):
        diff_timestamps = np.setdiff1d(
            all_timestamps, constellation_timestamps[i], assume_unique=True
        )
        diff_bands = np.setdiff1d(all_bands, constellation_bands[i], assume_unique=True)
        zero_arr_time = da.full(
            (len(diff_timestamps), *c_da.shape[1:]),
            fill_value=np.nan,
            chunks=c_da.data.chunksize,
            dtype=c_da.dtype,
        )

        zero_data_arr = xr.DataArray(
            zero_arr_time,
            coords={"revisits": diff_timestamps, "bands": constellation_bands[i]},
            dims=["revisits", "bands", "height", "width"],
        )
        concat_arr = xr.concat([c_da, zero_data_arr], dim="revisits")

        zero_arr_band = da.full(
            (len(all_timestamps), len(diff_bands), *c_da.shape[2:]),
            fill_value=np.nan,
            chunks=c_da.data.chunksize,
            dtype=c_da.dtype,
        )
        zero_data_arr = xr.DataArray(
            zero_arr_band,
            coords={"revisits": all_timestamps, "bands": diff_bands},
            dims=["revisits", "bands", "height", "width"],
        )

        concat_arr = xr.concat([concat_arr, zero_data_arr], dim="bands")
        constellation_data_arrs[i] = concat_arr

    ds = {
        constellations[i].name: (
            ["revisits", "bands", "height", "width"],
            constellation_data_arrs[i].sortby(["revisits", "bands"]).data,
        )
        for i in range(len(constellations))
    }
    ds = xr.Dataset(
        ds,
        coords={"revisits": all_timestamps, "bands": sorted(all_bands)},
        attrs={"patch_files": constellation_attrs},
    )
    return ds


def load_virtual_datasets(
    constellation_regions: Dict[str, List[List[ConstellationData]]],
    date_range,
    fs_mapper=None,
):

    assert constellation_regions.get("data") is not None, "'data' key must be present"

    if date_range is not None:
        start_date = date_range[0]
        end_date = date_range[1]
    else:
        start_date, end_date = None, None

    data_virtual_datasets = [
        constellations_dataset(c_region, fs_mapper=fs_mapper)
        for c_region in constellation_regions["data"]
    ]

    label_virtual_datasets = None

    if constellation_regions.get("label") is not None:
        label_virtual_datasets = constellation_regions["label"]

        label_virtual_datasets = [
            constellations_dataset(
                c_region, data_path=f"mask/{c_region[0].bands[0]}", fs_mapper=fs_mapper
            )
            for c_region in constellation_regions["label"]
        ]
        label_virtual_datasets = label_virtual_datasets
        label_virtual_datasets = [
            ds.sel({"revisits": slice(start_date, end_date)})
            for ds in label_virtual_datasets
        ]

        data_virtual_datasets = [
            ds.sel({"revisits": label_virtual_datasets[i]["revisits"]})
            for i, ds in enumerate(data_virtual_datasets)
        ]

    data_virtual_datasets = [
        ds.sel({"revisits": slice(start_date, end_date)})
        for ds in data_virtual_datasets
    ]
    return data_virtual_datasets, label_virtual_datasets


def create_index_map(
    constellation_regions: Dict[str, List[List[ConstellationData]]],
    date_range: Tuple[str, str],
    patch_size: int,
    output: str,
):
    data_virtual_datasets, _ = load_virtual_datasets(
        constellation_regions, date_range=date_range
    )
    index_map = []
    for dataset_index in range(len(data_virtual_datasets)):
        timestamps = data_virtual_datasets[dataset_index]["revisits"].data
        for timestamp_index in timestamps:
            for i in range(
                data_virtual_datasets[dataset_index].dims["height"] // patch_size
            ):
                for j in range(
                    data_virtual_datasets[dataset_index].dims["width"] // patch_size
                ):
                    index_map.append((dataset_index, timestamp_index, i, j))

    if output:
        pd.DataFrame(index_map).to_csv(output, index=False, header=None)
    return index_map
