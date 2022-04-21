import os
from typing import Union

import pandas as pd
from google.cloud import bigquery
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


@app.get("/water/{area_id}/timeseries")
def get_timeseries(
    area_id: int,
    password: str,
    constellation: str = "all",
    for_web: bool = True,  # set to False to get a dict that pandas likes
    resample: str = "30D",
) -> Union[dict[str, list[dict]], list[dict]]:
    """
    Get the timeseries for a single lake.
    Gets a single area value for each date, using the date embedded in run_id
    to get the latest run.

    If constellation=all, all constellations are returned.

    If for_web=False: return a single list of dicts like {date: ..., area: ...}
    if for_web=True: returns a dict of lists, keyed by constellation.
    """

    if password != "helsinki":
        return {"error": "not authenticated"}

    filt_cons = (
        "" if constellation == "all" else f"AND constellation = '{constellation}'"
    )
    query = f"""
    SELECT DISTINCT
      date,
      constellation,
      FIRST_VALUE (area) OVER w AS area,
    FROM `oxeo-main.water.water_ts`
    WHERE area_id = {area_id}
    {filt_cons}
    AND area > 0
    WINDOW w AS (
      PARTITION BY area_id, constellation, date
      ORDER BY SPLIT(run_id, '_')[OFFSET(1)] DESC
    )
    ORDER BY date
    """

    client = bigquery.Client()
    job = client.query(query)
    data = [dict(row) for row in job]
    if len(data) == 0:
        return {"error": "no data for area_id/constellation"}

    if for_web:
        df = (
            pd.DataFrame(data)
            .assign(date=lambda x: pd.to_datetime(x.date))
            .groupby("constellation")
            .resample(resample, on="date")
            .mean()
            .reset_index()
            .assign(date=lambda x: x.date.astype(str))
            .fillna({"area": 0})
            .assign(area=lambda x: x.area.astype(int))
        )
        return {
            constellation: (
                df.loc[df.constellation == constellation, ["date", "area"]].to_dict(
                    orient="records"
                )
            )
            for constellation in df.constellation.unique()
        }
    else:
        return data
