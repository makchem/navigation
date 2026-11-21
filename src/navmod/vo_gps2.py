"""Experimental SITL-only adapter: relative VO -> georeferenced GPS2 position.

This is NOT a calibrated GPS replacement. Uses GPS1 (when present) or a
one-time map-matcher fix to anchor a relative VO segment. Never reads truth.
Does not send MAVLink; bridge controls the injection and altitude source.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from navmod.georef import MapGeoref, enu_to_latlon, latlon_to_enu


@dataclass(frozen=True)
class VOGpsFix:
    lat: float
    lon: float
    # Conservative placeholder (NOT a calibrated uncertainty).
    reported_accuracy_m: float
    vo_sigma_m: float
    anchor_source: str
    segment_id: int


class VOGps2Adapter:
    """Convert existing VO diagnostic rows into optional GPS2 fixes.

    Fixes are only emitted for OK rows from a segment with a known anchor.
    The anchor is refreshed by GPS1 before the outage, or established once
    using an independent matcher fix if GPS1 has already disappeared.
    A LOST row or segment change invalidates the old anchor.
    """

    def __init__(self, georef: MapGeoref, *, max_sigma_m: float = 30.0,
                 max_speed_mps: float = 65.0):
        self.georef = georef
        self.max_sigma_m = max_sigma_m
        self.max_speed_mps = max_speed_mps
        self.reset()

    def reset(self) -> None:
        self.segment_id: int | None = None
        self.anchor_offset: tuple[float, float] | None = None
        self.anchor_source: str | None = None

    @staticmethod
    def _finite(v) -> float | None:
        try:
            val = float(v)
        except (TypeError, ValueError):
            return None
        return val if math.isfinite(val) else None

    def update(self, *, vo_row: dict | None, gps, matcher_fix) -> VOGpsFix | None:
        if vo_row is None:
            return None
        try:
            segment = int(vo_row["segment_id"])
        except (KeyError, ValueError, TypeError):
            return None
        if self.segment_id != segment:
            self.segment_id = segment
            self.anchor_offset = None
            self.anchor_source = None

        status = vo_row.get("status")
        if status == "LOST":
            self.anchor_offset = None
            self.anchor_source = None
            return None
        if status not in {"INIT", "OK"}:
            return None

        e = self._finite(vo_row.get("position_east_m"))
        n = self._finite(vo_row.get("position_north_m"))
        if e is None or n is None:
            return None

        # GPS1 bootstrap when it exists, without using SITL truth coordinates.
        if gps.available and all(map(math.isfinite, (gps.lat, gps.lon))):
            ge, gn = latlon_to_enu(self.georef, gps.lat, gps.lon)
            self.anchor_offset = (ge - e, gn - n)
            self.anchor_source = "GPS1"
        elif self.anchor_offset is None and matcher_fix.ok and all(
            map(math.isfinite, (matcher_fix.lat, matcher_fix.lon))
        ):
            # An initial camera-to-map fix can anchor VO without GPS1.
            ge, gn = latlon_to_enu(self.georef, matcher_fix.lat, matcher_fix.lon)
            self.anchor_offset = (ge - e, gn - n)
            self.anchor_source = "MATCHER"

        if status != "OK" or self.anchor_offset is None:
            return None
        sigma = self._finite(vo_row.get("sigma_m"))
        ve = self._finite(vo_row.get("ve_mps"))
        vn = self._finite(vo_row.get("vn_mps"))
        if (sigma is None or ve is None or vn is None or
                sigma < 0 or sigma > self.max_sigma_m or
                math.hypot(ve, vn) > self.max_speed_mps):
            return None

        east = self.anchor_offset[0] + e
        north = self.anchor_offset[1] + n
        lat, lon = enu_to_latlon(self.georef, east, north)
        if not (self.georef.bottom_right_lat <= lat <= self.georef.top_left_lat
                and self.georef.top_left_lon <= lon <= self.georef.bottom_right_lon):
            return None

        # Deliberately conservative, but still NOT a calibrated GPS error.
        accuracy = max(25.0, sigma * 4.0)
        return VOGpsFix(lat, lon, accuracy, sigma, self.anchor_source, segment)
