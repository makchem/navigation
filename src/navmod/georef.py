"""Coordinate conversions: pixel <-> local metres (ENU) <-> lat/lon.

One well-tested conversion layer. All other modules go through this
so there is exactly one place where (h, w) vs (w, h) conventions live.

Coordinate systems:
  pixel:  (px_x, px_y) — origin top-left, x right, y down (OpenCV/NumPy)
  ENU:    (east_m, north_m) — metres from the map origin (top-left corner)
          east = +x, north = -y (north is up, but pixel y grows downward)
  WGS84:  (lat, lon) in degrees
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyproj import Transformer


@dataclass(frozen=True)
class MapGeoref:
    """Georeference for a raster map image.

    Defined by four corner lat/lons and the image size in pixels.
    Assumes the map is north-up and uses a simple affine (no rotation
    or shear in the georef itself — the map image is already aligned).
    """
    top_left_lat: float
    top_left_lon: float
    bottom_right_lat: float
    bottom_right_lon: float
    image_width_px: int
    image_height_px: int

    def __post_init__(self):
        if self.top_left_lat <= self.bottom_right_lat:
            raise ValueError("top_left_lat must be north of bottom_right_lat")
        if self.image_width_px <= 0 or self.image_height_px <= 0:
            raise ValueError("image dimensions must be positive")

    @property
    def ground_width_m(self) -> float:
        """East-west extent in metres (at the map's mid-latitude)."""
        mid_lat = (self.top_left_lat + self.bottom_right_lat) / 2
        return _haversine_m(mid_lat, self.top_left_lon,
                            mid_lat, self.bottom_right_lon)

    @property
    def ground_height_m(self) -> float:
        """North-south extent in metres."""
        return _haversine_m(self.top_left_lat, self.top_left_lon,
                            self.bottom_right_lat, self.top_left_lon)

    @property
    def gsd_x(self) -> float:
        """Metres per pixel in the east-west direction."""
        return self.ground_width_m / self.image_width_px

    @property
    def gsd_y(self) -> float:
        """Metres per pixel in the north-south direction."""
        return self.ground_height_m / self.image_height_px


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two WGS84 points."""
    R = 6_371_000.0
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(R * 2 * np.arcsin(np.sqrt(a)))


def px_to_enu(georef: MapGeoref, px_x: float, px_y: float) -> tuple[float, float]:
    """Pixel (x, y) -> ENU (east_m, north_m) relative to the map top-left."""
    east_m = px_x * georef.gsd_x
    north_m = -px_y * georef.gsd_y   # pixel y down, north up
    return east_m, north_m


def enu_to_px(georef: MapGeoref, east_m: float, north_m: float) -> tuple[float, float]:
    """ENU (east_m, north_m) -> pixel (x, y)."""
    px_x = east_m / georef.gsd_x
    px_y = -north_m / georef.gsd_y
    return px_x, px_y


def px_to_latlon(georef: MapGeoref, px_x: float, px_y: float) -> tuple[float, float]:
    """Pixel (x, y) -> (lat, lon) via linear interpolation in the georef grid."""
    frac_x = px_x / georef.image_width_px
    frac_y = px_y / georef.image_height_px
    lat = georef.top_left_lat + frac_y * (georef.bottom_right_lat - georef.top_left_lat)
    lon = georef.top_left_lon + frac_x * (georef.bottom_right_lon - georef.top_left_lon)
    return lat, lon


def latlon_to_px(georef: MapGeoref, lat: float, lon: float) -> tuple[float, float]:
    """(lat, lon) -> pixel (x, y) via linear interpolation."""
    frac_x = (lon - georef.top_left_lon) / (georef.bottom_right_lon - georef.top_left_lon)
    frac_y = (lat - georef.top_left_lat) / (georef.bottom_right_lat - georef.top_left_lat)
    px_x = frac_x * georef.image_width_px
    px_y = frac_y * georef.image_height_px
    return px_x, px_y


def latlon_to_enu(georef: MapGeoref, lat: float, lon: float) -> tuple[float, float]:
    """(lat, lon) -> ENU (east_m, north_m) relative to map top-left."""
    px_x, px_y = latlon_to_px(georef, lat, lon)
    return px_to_enu(georef, px_x, px_y)


def enu_to_latlon(georef: MapGeoref, east_m: float, north_m: float) -> tuple[float, float]:
    """ENU (east_m, north_m) -> (lat, lon)."""
    px_x, px_y = enu_to_px(georef, east_m, north_m)
    return px_to_latlon(georef, px_x, px_y)


def distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in metres between two WGS84 points (public API)."""
    return _haversine_m(lat1, lon1, lat2, lon2)
