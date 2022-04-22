from typing import Optional, TypeVar, Union

from datetime import date

from pydantic import BaseModel, Field, conlist


class Message(BaseModel):
    message: str = Field(..., example="Health OK!")


class HTTPError(BaseModel):
    detail: str

    class Config:
        schema_extra = {
            "example": {"detail": "HTTPException raised."},
        }


class Basin(BaseModel):
    id: int
    ndvi_avg: float


class Value(BaseModel):
    date: date
    area: float


class TimeSeries(BaseModel):
    __root__: list[Value]


class Constellations(BaseModel):
    __root__: dict[str, TimeSeries]


class PandasValue(BaseModel):
    date: date
    constellation: str
    area: float


class PandasTimeSeries(BaseModel):
    __root__: list[PandasValue]


class Lake(BaseModel):
    id: int
    name: str
    constellations: Constellations


class Asset(BaseModel):
    name: str
    id: str
    materiality: int
    lakes: list[Lake]
    basins: list[Basin]
    geom: dict


class Company(BaseModel):
    name: str
    assets: list[Asset]


# The geometry types below borrow heavily from
# https://github.com/developmentseed/geojson-pydantic
Point = tuple[float, float]
LinearRing = conlist(Point, min_items=4)
PolygonCoords = conlist(LinearRing, min_items=1)
MultiPolygonCoords = conlist(PolygonCoords, min_items=1)
BBox = tuple[float, float, float, float]  # 2D bbox
Props = TypeVar("Props", bound=dict)


class Geometry(BaseModel):
    type: str = Field(..., example="Polygon")
    coordinates: Union[PolygonCoords, MultiPolygonCoords] = Field(
        ..., example=[[[1, 3], [2, 2], [4, 4], [1, 3]]]
    )


class Feature(BaseModel):
    type: str = Field("Feature", const=True)
    geometry: Geometry
    properties: Optional[Props]
    id: Optional[str]
    bbox: Optional[BBox]


class FeatureCollection(BaseModel):
    type: str = Field("FeatureCollection", const=True)
    features: list[Feature]

    def __iter__(self):
        """iterate over features"""
        return iter(self.features)

    def __len__(self):
        """return features length"""
        return len(self.features)

    def __getitem__(self, index):
        """get feature at a given index"""
        return self.features[index]
