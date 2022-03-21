import os

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from oxeo.core.models.data import get_water_geoms
from oxeo.core.models.tile import get_tiles, tile_to_geom

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


@app.get("/")
def test():
    return "API is running"


@app.get("/water/{area_id}/tiles/geom")
def get_lake_tile_geoms(area_id: int):
    gdf = get_water_geoms(
        [area_id],
        db_user=DB_USER,
        db_password=DB_PW,
        db_host=DB_HOST,
    )
    tiles = get_tiles(gdf)
    geoms = pd.concat([tile_to_geom(t) for t in tiles])
    geoms["tile"] = [t.id for t in tiles]
    return geoms.__geo_interface__
