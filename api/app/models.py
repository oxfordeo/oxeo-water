from pydantic import BaseModel


class Basin(BaseModel):
    id: int
    ndvi_avg: float


class Value(BaseModel):
    date: str
    area: float


class Lake(BaseModel):
    id: int
    name: str
    area: dict
    timeseries: list[Value]


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
