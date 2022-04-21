from typing import Dict, List, Tuple, Union

import geopandas as gpd
import numpy as np
import sentinelhub
import zarr
from attr import define
from pyproj import CRS
from shapely.geometry import MultiPolygon, Polygon

from oxeo.core.logging import logger
from oxeo.core.utils import get_band_list, get_transform_function


@define(frozen=True)
class Tile:
    zone: int
    row: str
    min_x: int
    min_y: int
    max_x: int
    max_y: int
    epsg: str

    @property
    def id(self) -> str:
        return f"{self.zone}_{self.row}_{self.bbox_size_x}_{self.xloc}_{self.yloc}"

    @property
    def xloc(self) -> int:
        return int(self.min_x / self.bbox_size_x)

    @property
    def yloc(self) -> int:
        return int(self.min_y / self.bbox_size_y)

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        return (self.min_x, self.min_y, self.max_x, self.max_y)

    @property
    def bbox_wgs84(self):
        reproj_src_wgs = get_transform_function(str(self.epsg), "WGS84")
        return (
            *reproj_src_wgs(self.min_x, self.min_y),
            *reproj_src_wgs(self.max_x, self.max_y),
        )

    @property
    def bbox_size_x(self) -> int:  # in metres
        return int(self.max_x - self.min_x)

    @property
    def bbox_size_y(self) -> int:  # in metres
        return int(self.max_y - self.min_y)

    @property
    def bbox_size(self) -> Tuple[int, int]:  # in metres
        return (self.bbox_size_x, self.bbox_size_y)

    def contains(self, other) -> bool:
        return (
            self.epsg == other.epsg
            and self.bbox[0] <= other.bbox[0]
            and self.bbox[1] <= other.bbox[1]
            and self.bbox[2] >= other.bbox[2]
            and self.bbox[3] >= other.bbox[3]
        )


@define(frozen=True)
class TilePath:
    tile: Tile
    constellation: str
    root: str = "gs://oxeo-water/prod"

    @property
    def path(self):
        return f"{self.root}/{self.tile.id}/{self.constellation}"

    @property
    def timestamps_path(self):
        return f"{self.path}/timestamps"

    @property
    def data_path(self):
        return f"{self.path}/data"

    @property
    def mask_path(self):
        return f"{self.path}/mask"

    @property
    def metadata_path(self):
        return f"{self.path}/metadata"


def tile_from_id(id: str) -> Tile:
    zone, row, bbox_size_x, xloc, yloc = id.split("_")
    bbox_size_x, xloc, yloc = int(bbox_size_x), int(xloc), int(yloc)
    bbox_size_y = bbox_size_x
    min_x = xloc * bbox_size_x
    min_y = yloc * bbox_size_y
    max_x = min_x + bbox_size_x
    max_y = min_y + bbox_size_y
    south = row < "N"
    epsg = CRS.from_dict({"proj": "utm", "zone": zone, "south": south}).to_epsg()
    return Tile(
        zone=int(zone),
        row=row,
        min_x=min_x,
        min_y=min_y,
        max_x=max_x,
        max_y=max_y,
        epsg=epsg,
    )


def get_patch_size(patch_paths: List[TilePath]) -> int:  # in pixels
    # TODO: Probably unnecessary to load all patches for this,
    # could just assume they're the same size
    sizes = []
    logger.info("Checking arr sizes")
    for patch in patch_paths:
        arr_path = f"{patch.path}/data"
        logger.debug(f"Loading to check size {arr_path=}")
        z = zarr.open(arr_path, "r")
        x, y = z.shape[2:]
        assert x == y, "Must use square patches"
        sizes.append(x)
    assert len(set(sizes)) == 1, "All sizes must be the same and not empty."
    return sizes[0]


def get_tile_size(tiles: List[Tile]) -> int:  # in metres
    sizes = []
    for tile in tiles:
        x, y = tile.bbox_size_x, tile.bbox_size_y
        assert x == y, "Must use square tiles"
        sizes.append(x)
    assert len(set(sizes)) == 1, "All sizes must be the same"
    return sizes[0]


def make_paths(tiles, constellations, root_dir) -> List[TilePath]:
    return [
        TilePath(tile=tile, constellation=cons, root=root_dir)
        for tile in tiles
        for cons in constellations
    ]


def split_region_in_utm_tiles(
    region: Union[Polygon, MultiPolygon],
    crs: sentinelhub.CRS = sentinelhub.CRS.WGS84,
    bbox_size: int = 10000,
    **kwargs,
) -> List[Tile]:
    """Split a given geometry in squares measured in meters.
    It splits the region in utm grid and the convert back to given crs.

    Args:
        region (UnionList[shapely.geometry.Polygon, shapely.geometry.MultiPolygon]): The region to split from
        bbox_size (int): bbox size in meters

    Returns:
        [List[Tile]]: The Tiles representing each of the boxes
    """
    utm_splitter = sentinelhub.UtmGridSplitter([region], crs, bbox_size)
    crs_bboxes = utm_splitter.get_bbox_list()
    info_bboxes = utm_splitter.get_info_list()

    tiles = [
        Tile(
            zone=info["utm_zone"],
            row=info["utm_row"],
            min_x=box.min_x,
            min_y=box.min_y,
            max_x=box.max_x,
            max_y=box.max_y,
            epsg=box.crs.epsg,
        )
        for info, box in zip(info_bboxes, crs_bboxes)
    ]

    return tiles


def get_tiles(
    geom: Union[Polygon, MultiPolygon, gpd.GeoSeries, gpd.GeoDataFrame]
) -> list[Tile]:
    try:
        geom = geom.unary_union
    except AttributeError:
        pass
    return split_region_in_utm_tiles(region=geom, bbox_size=10000)


def get_all_paths(
    gdf: gpd.GeoDataFrame,
    constellations: list[str],
    root_dir: str = "gs://oxeo-water/prod",
) -> list[TilePath]:
    all_tiles = get_tiles(gdf)
    all_tilepaths = make_paths(all_tiles, constellations, root_dir)
    logger.info(
        f"All tiles for the supplied geometry: {[t.path for t in all_tilepaths]}"
    )
    return all_tilepaths


def load_tile_as_dict(
    fs_mapper,
    tile_path: TilePath,
    masks: Tuple[str, ...] = (),
    revisit: slice = None,
    bands: Tuple[str, ...] = None,
) -> Dict[str, np.ndarray]:
    """Loads a tile path as dictionary where keys are: image, mask_1, ..., mask_n

    Args:
        fs_mapper (_type_): a file system mapper (can be identity function if local file)
        tile_path (TilePath): the tile path to load
        masks (Tuple[str, ...], optional): a tuple of masks to load from zarr storage. Defaults to ().
        revisit (slice, optional): slice of revisits to load. Defaults to None.
        bands (Tuple[str, ...], optional): a tuple of bands to load. Defaults to None.

    Returns:
        Dict[str, np.ndarray]: a dictionary with an "image" key and "mask_1,...mask_n" keys.
                          and ndarray as values
    """
    if bands is not None:
        band_common_names = get_band_list(tile_path.constellation)
        band_indices = [band_common_names.index(b) for b in bands]
    else:
        band_common_names = get_band_list(tile_path.constellation)
        band_indices = list(range(0, len(band_common_names)))

    sample = {}
    arr = zarr.open_array(fs_mapper(tile_path.data_path), mode="r")
    logger.info(f"{arr.shape=}; {revisit=}, {band_indices=}")
    arr = arr.oindex[revisit, band_indices].astype(np.int16)
    logger.info(f"{arr.shape=}")
    for mask in masks:
        mask_arr = zarr.open_array(
            fs_mapper(f"{tile_path.mask_path}/{mask}"), mode="r"
        )[revisit].astype(np.int8)
        mask_arr = mask_arr[np.newaxis, ...]
        sample[mask] = mask_arr
    sample["image"] = arr
    return sample


def tile_to_geom(tile: Union[Tile, str]) -> gpd.GeoDataFrame:
    """Get the geom for a single tile as a GeoDataFrame."""
    if isinstance(tile, str):
        tile = tile_from_id(tile)

    bbox = (tile.min_x, tile.min_y, tile.max_x, tile.max_y)
    bbox = sentinelhub.BBox(bbox, crs=CRS.from_epsg(tile.epsg))
    bbox_poly = Polygon(bbox.transform(sentinelhub.CRS.WGS84).get_polygon())
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_poly], crs=CRS.from_epsg(4326))
    return bbox_gdf
