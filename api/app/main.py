import os
import secrets

import pandas as pd
from fastapi import Depends, FastAPI, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from oxeo.core.models.data import get_water_geoms
from oxeo.core.models.tile import get_tiles, tile_to_geom

from .data import get_timeseries
from .models import (
    Message,
    Lake,
    FeatureCollection,
    PandasTimeSeries,
    HTTPError,
    Company,
    Asset,
    Basin,
)

security = HTTPBasic()

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

USERNAME = os.environ["USERNAME"]
PASSWORD = os.environ["PASSWORD"]

DB_USER = os.environ.get("PG_DB_USER")
DB_PW = os.environ.get("PG_DB_PW")
DB_HOST = os.environ.get("PG_DB_HOST")


def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, USERNAME)
    correct_password = secrets.compare_digest(credentials.password, PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return True


requires_auth = [Depends(authenticate)]

other_responses = {
    401: {"model": HTTPError, "description": "Unauthorized"},
    400: {"model": HTTPError, "description": "Parameter error"},
}


@app.get("/healthz", response_model=Message, status_code=status.HTTP_200_OK)
def health():
    return {"message": "Health OK!"}


# TODO
# Replace these with real queries to somewhere
fake_company = Company(id="123", name="Fake Corp", assets=["mine1", "smelt1"])
fake_asset1 = Asset(id="mine1", name="Big Copper Mine", lakes=[-9050201])
fake_asset2 = Asset(id="smelt1", name="Lil Smelter", lakes=[-5867868], basins=[453809])
fake_assets = [fake_asset1, fake_asset2]
fake_basin1 = Basin(pfaf=453809, ndvi_avg=0.9)
fake_basin2 = Basin(pfaf=453808, ndvi_avg=0.6)
fake_basins = [fake_basin1, fake_basin2]
fake_lake1 = Lake(
    area_id=-9050201,
    name="krs",
    constellations={"s2": [{"date": "2000-01-01", "area": 100}]},
)
fake_lake2 = Lake(
    area_id=-5867868,
    name="kabini",
    constellations={"ls8": [{"date": "2000-01-01", "area": 50}]},
)
fake_lakes = [fake_lake1, fake_lake2]


@app.get("/companies", response_model=list[Company])
def companies():
    return [fake_company]


@app.get("/companies/{id}", response_model=Company)
def company(id: str):
    return fake_company


@app.get("/assets", response_model=list[Asset])
def assets():
    return fake_assets


@app.get("/asset/{id}", response_model=Asset)
def asset(id: str):
    return [x for x in fake_assets if x.id == id][0]


@app.get("/basins", response_model=list[Basin])
def basins():
    return fake_basins


@app.get("/basins/{pfaf}", response_model=Basin)
def basin(pfaf: int):
    return [x for x in fake_basins if x.pfaf == pfaf][0]


@app.get("/lakes", response_model=list[Lake])
def lakes():
    return fake_lakes


@app.get("/lakes/{area_id}", response_model=Lake)
def lake(area_id: int):
    return [x for x in fake_lakes if x.area_id == area_id][0]


@app.get(
    "/water/{area_id}/tiles/geom",
    dependencies=requires_auth,
    response_model=FeatureCollection,
    responses=other_responses,
)
def get_lake_tile_geoms(area_id: int):
    gdf = get_water_geoms(
        [area_id], db_user=DB_USER, db_password=DB_PW, db_host=DB_HOST
    )
    tiles = get_tiles(gdf)
    geoms = pd.concat([tile_to_geom(t) for t in tiles])
    geoms["tile"] = [t.id for t in tiles]
    return geoms.__geo_interface__


@app.get(
    "/water/{area_id}/timeseries",
    dependencies=requires_auth,
    response_model=Lake,
    responses=other_responses,
)
def timeseries(
    area_id: int,
    constellation: str = "all",
    resample: str = "30D",
):
    """
    Get the timeseries for a single lake.
    Gets a single area value for each date, using the date embedded in run_id
    to get the latest run.

    If constellation=all, all constellations are returned.
    """

    data = get_timeseries(area_id, constellation, resample, for_web=True)
    return data


@app.get(
    "/water/{area_id}/timeseries_pandas",
    dependencies=requires_auth,
    response_model=PandasTimeSeries,
    responses=other_responses,
)
def timeseries_pandas(
    area_id: int,
    constellation: str = "all",
    resample: str = "30D",
):
    """
    Same as `timeseries()` but gets a dict that Pandas can easily ingest.
    """

    data = get_timeseries(area_id, constellation, resample, for_web=False)
    return data
