import os

import pandas as pd
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware

from oxeo.core.models.data import get_water_geoms
from oxeo.core.models.tile import get_tiles, tile_to_geom

from .data import get_timeseries
from .models import Message, Lake, FeatureCollection

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_USER = os.environ.get("PG_DB_USER")
DB_PW = os.environ.get("PG_DB_PW")
DB_HOST = os.environ.get("PG_DB_HOST")


@app.get("/healthz", response_model=Message, status_code=status.HTTP_200_OK)
def health():
    return {"message": "Health OK!"}


@app.get("/water/{area_id}/tiles/geom", response_model=FeatureCollection)
def get_lake_tile_geoms(area_id: int):
    gdf = get_water_geoms(
        [area_id], db_user=DB_USER, db_password=DB_PW, db_host=DB_HOST
    )
    tiles = get_tiles(gdf)
    geoms = pd.concat([tile_to_geom(t) for t in tiles])
    geoms["tile"] = [t.id for t in tiles]
    return geoms.__geo_interface__


@app.get("/water/{area_id}/timeseries")
def timeseries(
    area_id: int,
    password: str,
    constellation: str = "all",
    resample: str = "30D",
) -> Lake:
    """
    Get the timeseries for a single lake.
    Gets a single area value for each date, using the date embedded in run_id
    to get the latest run.

    If constellation=all, all constellations are returned.
    """

    if password != "helsinki":
        return {"error": "not authenticated"}

    data = get_timeseries(area_id, constellation, resample, for_web=True)
    return data


@app.get("/water/{area_id}/timeseries_pandas")
def timeseries_pandas(
    area_id: int,
    password: str,
    constellation: str = "all",
    resample: str = "30D",
) -> dict[str, list[dict]]:
    """
    Same as `timeseries()` but gets a dict that Pandas can easily ingest.
    """

    if password != "helsinki":
        return {"error": "not authenticated"}

    data = get_timeseries(area_id, constellation, resample, for_web=False)
    return data
