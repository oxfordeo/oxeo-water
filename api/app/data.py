import pandas as pd
from google.cloud import bigquery

from .models import Lake


def get_timeseries(
    area_id: int,
    constellation: str,
    resample: str = "30D",
    for_web: bool = True,
) -> Lake:
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
