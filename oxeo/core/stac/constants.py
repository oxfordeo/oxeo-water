from enum import Enum
from typing import Any, Dict

import pystac
from pystac import ProviderRole
from pystac.extensions.eo import Band
from pystac.link import Link

USWEST_URL = "https://services-uswest2.sentinel-hub.com"
ELEMENT84_URL = "https://earth-search.aws.element84.com/v0"
LANDSATLOOK_URL = "https://landsatlook.usgs.gov/stac-server"

LANDSAT_SEARCH_PARAMS = {
    "query": {
        "platform": {"in": ["LANDSAT_5", "LANDSAT_7", "LANDSAT_8"]},
        "landsat:collection_category": {"in": ["T1"]},
    }
}

INSPIRE_METADATA_ASSET_KEY = "inspire-metadata"
SAFE_MANIFEST_ASSET_KEY = "safe-manifest"
PRODUCT_METADATA_ASSET_KEY = "product-metadata"

SENTINEL_LICENSE = Link(
    rel="license",
    target="https://sentinel.esa.int/documents/"
    + "247904/690755/Sentinel_Data_Legal_Notice",
)

ACQUISITION_MODES = [
    "Stripmap (SM)",
    "Interferometric Wide Swath (IW)",
    "Extra Wide Swath (EW)",
    "Wave (WV)",
]
SENTINEL_CONSTELLATION = "Sentinel 1"

SENTINEL_PROVIDER = pystac.Provider(
    name="ESA",
    roles=[
        ProviderRole.PRODUCER,
        ProviderRole.PROCESSOR,
        ProviderRole.LICENSOR,
    ],
    url="https://earth.esa.int/web/guest/home",
)

SAFE_MANIFEST_ASSET_KEY = "safe-manifest"

SENTINEL_POLARISATIONS = {
    "vh": Band.create(
        name="VH",
        description="vertical transmit and horizontal receive",
    ),
    "hh": Band.create(
        name="HH",
        description="horizontal transmit and horizontal receive",
    ),
    "hv": Band.create(
        name="HV",
        description="horizontal transmit and vertical receive",
    ),
    "vv": Band.create(
        name="VV",
        description="vertical transmit and vertical receive",
    ),
}

SENTINEL_LICENSE = Link(
    rel="license",
    target="https://sentinel.esa.int/documents/"
    + "247904/690755/Sentinel_Data_Legal_Notice",
)


class Sensor(Enum):
    MSS = "M"
    TM = "T"
    ETM = "E"
    OLI_TIRS = "C"


LANDSAT_EXTENSION_SCHEMA = (
    "https://landsat.usgs.gov/stac/landsat-extension/v1.1.1/schema.json"
)
CLASSIFICATION_EXTENSION_SCHEMA = (
    "https://stac-extensions.github.io/classification/v1.0.0/schema.json"  # noqa
)
USGS_API = "https://landsatlook.usgs.gov/stac-server"
USGS_BROWSER_C2 = "https://landsatlook.usgs.gov/stac-browser/collection02"
USGS_C2L1 = "landsat-c2l1"
USGS_C2L2_SR = "landsat-c2l2-sr"
USGS_C2L2_ST = "landsat-c2l2-st"
COLLECTION_IDS = ["landsat-c2-l1", "landsat-c2-l2"]

SENSORS: Dict[str, Any] = {
    "MSS": {
        "instruments": ["mss"],
        "doi": "10.5066/P9AF14YV",
        "doi_title": "Landsat 1-5 MSS Collection 2 Level-1",
        "reflective_gsd": 79,
    },
    "TM": {
        "instruments": ["tm"],
        "doi": "10.5066/P9IAXOVV",
        "doi_title": "Landsat 4-5 TM Collection 2 Level-2",
        "reflective_gsd": 30,
        "thermal_gsd": 120,
    },
    "ETM": {
        "instruments": ["etm+"],
        "doi": "10.5066/P9C7I13B",
        "doi_title": "Landsat 7 ETM+ Collection 2 Level-2",
        "reflective_gsd": 30,
        "thermal_gsd": 60,
    },
    "OLI_TIRS": {
        "instruments": ["oli", "tirs"],
        "doi": "10.5066/P9OGBGM6",
        "doi_title": "Landsat 8-9 OLI/TIRS Collection 2 Level-2",
        "reflective_gsd": 30,
        "thermal_gsd": 100,
    },
}

L8_PLATFORM = "landsat-8"
L8_INSTRUMENTS = ["oli", "tirs"]

OLD_L8_EXTENSION_SCHEMA = "https://landsat.usgs.gov/stac/landsat-extension/schema.json"
L8_EXTENSION_SCHEMA = (
    "https://landsat.usgs.gov/stac/landsat-extension/v1.1.0/schema.json"
)
L8_ITEM_DESCRIPTION = "Landsat Collection 2 Level-2 Surface Reflectance Product"

L8_SR_BANDS = {
    "SR_B1": Band(
        {
            "name": "SR_B1",
            "common_name": "coastal",
            "gsd": 30,
            "center_wavelength": 0.44,
            "full_width_half_max": 0.02,
        }
    ),
    "SR_B2": Band(
        {
            "name": "SR_B2",
            "common_name": "blue",
            "gsd": 30,
            "center_wavelength": 0.48,
            "full_width_half_max": 0.06,
        }
    ),
    "SR_B3": Band(
        {
            "name": "SR_B3",
            "common_name": "green",
            "gsd": 30,
            "center_wavelength": 0.56,
            "full_width_half_max": 0.06,
        }
    ),
    "SR_B4": Band(
        {
            "name": "SR_B4",
            "common_name": "red",
            "gsd": 30,
            "center_wavelength": 0.65,
            "full_width_half_max": 0.04,
        }
    ),
    "SR_B5": Band(
        {
            "name": "SR_B5",
            "common_name": "nir08",
            "gsd": 30,
            "center_wavelength": 0.86,
            "full_width_half_max": 0.03,
        }
    ),
    "SR_B6": Band(
        {
            "name": "SR_B6",
            "common_name": "swir16",
            "gsd": 30,
            "center_wavelength": 1.6,
            "full_width_half_max": 0.08,
        }
    ),
    "SR_B7": Band(
        {
            "name": "SR_B7",
            "common_name": "swir22",
            "gsd": 30,
            "center_wavelength": 2.2,
            "full_width_half_max": 0.2,
        }
    ),
}

L8_SP_BANDS = {
    # L2SP only bands
    #  ST_B10 Note:
    # Changed common_name from UGSG STAC - should be lwir11 based on wavelength
    # Also, resolution at sensor is 100m, even though the raster is 30m pixel width/height.
    "ST_B10": Band(
        {
            "name": "ST_B10",
            "common_name": "lwir11",
            "gsd": 100.0,
            "center_wavelength": 10.9,
            "full_width_half_max": 0.8,
        }
    ),
    "ST_ATRAN": Band(
        {"name": "ST_ATRAN", "description": "atmospheric transmission", "gsd": 30}
    ),
    "ST_CDIST": Band(
        {"name": "ST_CDIST", "description": "distance to nearest cloud", "gsd": 30}
    ),
    "ST_DRAD": Band(
        {"name": "ST_DRAD", "description": "downwelled radiance", "gsd": 30}
    ),
    "ST_URAD": Band({"name": "ST_URAD", "description": "upwelled radiance", "gsd": 30}),
    "ST_TRAD": Band({"name": "ST_TRAD", "description": "thermal radiance", "gsd": 30}),
    "ST_EMIS": Band({"name": "ST_EMIS", "description": "emissivity", "gsd": 30}),
    "ST_EMSD": Band(
        {"name": "ST_EMSD", "description": "emissivity standard deviation", "gsd": 30}
    ),
}
