from typing import List, Union

import geopandas as gpd
from attr import define
from shapely.geometry import MultiPolygon, Polygon

from oxeo.core.models.tile import TilePath, get_tiles, make_paths


@define
class WaterBody:
    area_id: int
    name: str
    geometry: Union[Polygon, MultiPolygon]
    paths: List[TilePath]


def get_waterbodies(
    gdf: gpd.GeoDataFrame,
    constellations: List[str],
    root_dir: str = "gs://oxeo-water/prod",
) -> List[WaterBody]:
    waterbodies = []
    for water in gdf.to_dict(orient="records"):
        tiles = get_tiles(water["geometry"])
        waterbodies.append(
            WaterBody(
                **water,
                paths=make_paths(tiles, constellations, root_dir=root_dir),
            )
        )
    return waterbodies
