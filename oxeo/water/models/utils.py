import itertools
from datetime import datetime
from typing import List, Set

import numpy as np
import zarr


def parse_xy(patch_path: str):
    return [int(x) for x in patch_path.split("_")[-2:]]


def get_dates_in_common(
    patch_paths: List[str], constellation: str = "sentinel-2"
) -> List[datetime]:
    dates: Set[str] = set()
    for pp in patch_paths:
        date = zarr.open(f"gs://{pp}/{constellation}/timestamps", "r")[:]
        if len(dates) == 0:
            dates = set(date)
        else:
            dates = dates.intersection(date)
    return zarr_dates_to_datetime(dates)


def nearest(items: List[datetime], pivot: datetime):
    return min(items, key=lambda x: abs(x - pivot))


def zarr_dates_to_datetime(dates: Set[str]) -> List[datetime]:
    return [datetime.strptime(x[:10], "%Y-%m-%d") for x in sorted(list(dates))]


date_earliest = datetime(1900, 1, 1)
date_latest = datetime(2200, 1, 1)


def merge_masks(
    patch_paths: List[str],
    patch_size: int,
    start_date: datetime = date_earliest,
    end_date: datetime = date_latest,
    constellation: str = "sentinel-2",
    data: str = "weak_labels",
):
    xy = [parse_xy(pp) for pp in patch_paths]
    x, y = list(zip(*xy))
    # Hack to deal with changed tile naming system
    xy = [(p[0] - min(x), p[1] - min(y)) for p in xy]
    x, y = list(zip(*xy))
    max_x = max(x)
    max_y = max(y)

    common_dates = get_dates_in_common(patch_paths, constellation)
    nearest_start_date = nearest(common_dates, start_date)
    nearest_end_date = nearest(common_dates, end_date)

    start_date_index = common_dates.index(nearest_start_date)
    end_date_index = common_dates.index(nearest_end_date)

    # Create the fullmask array to contain the patches
    full_mask = np.zeros(
        (
            end_date_index - start_date_index,
            (max_y + 1) * patch_size,
            (max_x + 1) * patch_size,
        )
    )

    for i, pp in enumerate(patch_paths):
        # Get the dates for that patch
        dates = zarr.open(f"gs://{pp}/{constellation}/timestamps", "r")[:]
        dates = zarr_dates_to_datetime(dates)

        # This is tricky. Here I check for the dates that all patches share
        # I get the indices of those dates for each of the patches. So I can
        # Extract the correct patches at the end.
        date_indices = {d: index for index, d in enumerate(dates) if d in common_dates}

        keys = list(date_indices.keys())
        start_date_index = keys.index(nearest_start_date)
        end_date_index = keys.index(nearest_end_date)

        date_indices = dict(
            itertools.islice(date_indices.items(), start_date_index, end_date_index)
        )
        date_indices_vals = list(date_indices.values())

        # Once I have the indices I can get the patch and append it to the fullmask
        arr = zarr.open(f"gs://{pp}/{constellation}/{data}", "r")
        start_y = (max_y - xy[i][1]) * patch_size
        end_y = start_y + patch_size
        start_x = xy[i][0] * patch_size
        end_x = start_x + patch_size

        # I need to do this because zarr doesn't support indexing like numpy
        # So I bring all the dates until the last one and then a filter them.
        arr = arr[: date_indices_vals[-1] + 1].astype(np.uint8)
        full_mask[:, start_y:end_y, start_x:end_x] = arr[date_indices_vals, :]
        full_mask = full_mask.astype(np.uint8)

    # Return the mask and tha common dates in the given range.
    return full_mask, list(date_indices.keys())
