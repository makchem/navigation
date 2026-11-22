"""SITL-only matcher-first position fusion for the existing VO diagnostic stream.

The matcher supplies absolute map position when it succeeds.  In between matches
VO supplies relative displacement.  Late asynchronous matches are aligned to
the VO position of the IMAGE THEY MATCHED, not to the current frame.

No simulator ground truth enters this module.  Coordinates sent to EKF are local
NED with an origin seeded once from a GPS1 fix or an accepted map match.

This is experimental, not a safety-critical position estimator; covariance is
not calibrated and the current VO heading source may drift without GNSS.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from navmod.georef import MapGeoref, latlon_to_enu


@dataclass(frozen=True)
class MatchSnapshot:
    frame_id: int
    t_sim: float
    segment_id: int | None
    vo_e: float | None
    vo_n: float | None


@dataclass(frozen=True)
class MatchDecision:
    accepted: bool
    reason: str
    innovation_m: float | None = None


class MatcherFirstFusion:
    """Map matcher anchors/corrects pose; VO propagates it between matches.

    A reset of the VO segment invalidates the current offset. GPS1 can seed a
    fresh segment while available; without it we wait for a map match.
    Applying corrections at a limited rate avoids sudden jumps in ExternalNav.
    """

    def __init__(
        self,
        georef: MapGeoref,
        *,
        max_innovation_m: float = 120.0,
        correction_mps: float = 20.0,
        max_match_age_s: float = 5.0,
        recent_match_s: float = 1.5,
    ):
        self.georef = georef
        self.max_innovation_m = max_innovation_m
        self.correction_mps = correction_mps
        self.max_match_age_s = max_match_age_s
        self.recent_match_s = recent_match_s
        self.reset()

    def reset(self):
        self.origin: tuple[float, float] | None = None
        self.segment_id: int | None = None
        self.offset: tuple[float, float] | None = None
        self.target_offset: tuple[float, float] | None = None
        self._last_tick_t: float | None = None
        self.last_match_t: float | None = None
        self.last_match_frame: int | None = None
        self._new_match: bool = False
        self._pending_large_offset: tuple[float,float,float] | None = None

    def seed_origin(self, gps) -> None:
        """Freeze the map origin from GPS1 while on the ground/at startup."""
        if (self.origin is None and gps.available and
                all(math.isfinite(v) for v in (gps.lat, gps.lon))):
            self.origin = latlon_to_enu(self.georef, gps.lat, gps.lon)

    @staticmethod
    def _finite(value):
        try:
            f = float(value)
        except (TypeError, ValueError):
            return None
        return f if math.isfinite(f) else None

    def observe_frame(self, row: dict | None, gps) -> MatchSnapshot | None:
        if row is None:
            return None
        try:
            seg = int(row['segment_id'])
            frame_id = int(row['frame_id'])
            t = float(row['t_sim'])
        except (KeyError, TypeError, ValueError):
            return None
        if not math.isfinite(t):
            return None

        if self.segment_id != seg:
            self.segment_id = seg
            self.offset = None
            self.target_offset = None
            self._last_tick_t = None
            self.last_match_t = None
            self.last_match_frame = None
            self._new_match = False
            self._pending_large_offset = None

        status = row.get('status')
        e = self._finite(row.get('position_east_m'))
        n = self._finite(row.get('position_north_m'))
        if status == 'LOST' or e is None or n is None:
            if status == 'LOST':
                self.offset = None
                self.target_offset = None
            return MatchSnapshot(frame_id, t, seg, None, None)

        # GPS1 is only an initial reference, never a continuing correction.
        if gps.available and all(math.isfinite(v) for v in (gps.lat, gps.lon)):
            ge, gn = latlon_to_enu(self.georef, gps.lat, gps.lon)
            if self.origin is None:
                self.origin = (ge, gn)
            if self.offset is None:
                self.offset = (ge - e, gn - n)
                self.target_offset = self.offset
        return MatchSnapshot(frame_id, t, seg, e, n)

    def accept_match(self, fix, snap: MatchSnapshot, current_t: float) -> MatchDecision:
        if not getattr(fix, 'ok', False):
            return MatchDecision(False, 'NO_FIX')
        if (snap is None or snap.vo_e is None or snap.vo_n is None
                or snap.segment_id != self.segment_id):
            return MatchDecision(False, 'NO_VO_SNAPSHOT_OR_OLD_SEGMENT')
        lat = self._finite(getattr(fix, 'lat', None))
        lon = self._finite(getattr(fix, 'lon', None))
        if lat is None or lon is None:
            return MatchDecision(False, 'INVALID_MATCH_COORDINATES')
        age = current_t - snap.t_sim
        if not 0.0 <= age <= self.max_match_age_s:
            return MatchDecision(False, 'STALE_MATCH')
        ge, gn = latlon_to_enu(self.georef, lat, lon)
        if self.origin is None:
            self.origin = (ge, gn)

        # Prediction at the *capture* time, not the result delivery time.
        proposed_offset = (ge - snap.vo_e, gn - snap.vo_n)
        innovation = None
        if self.offset is not None:
            pred = (self.offset[0] + snap.vo_e, self.offset[1] + snap.vo_n)
            innovation = math.hypot(ge - pred[0], gn - pred[1])
            if innovation > self.max_innovation_m:
                # A single distant fix might be a false match. Two fixes
                # agreeing on the *same map-to-VO offset* can re-lock after
                # genuine, accumulated VO drift.
                pending = self._pending_large_offset
                consistent = (
                    pending is not None and 0 < snap.t_sim - pending[2] <= 8.0
                    and math.hypot(proposed_offset[0] - pending[0],
                                   proposed_offset[1] - pending[1]) <= 20.0
                )
                if not consistent:
                    self._pending_large_offset = (*proposed_offset, snap.t_sim)
                    return MatchDecision(False, 'MATCH_OUTLIER_WAIT_RECONFIRM', innovation)
        self._pending_large_offset = None
        self.target_offset = proposed_offset
        if self.offset is None:
            self.offset = self.target_offset
        self.last_match_t = snap.t_sim
        self.last_match_frame = snap.frame_id
        self._new_match = True
        return MatchDecision(True, 'MATCH_ACCEPTED', innovation)

    def fused_ned(self, row: dict | None) -> tuple[float, float, str, float] | None:
        """Return local N/E, source label and remaining correction residual.

        Only a fresh OK VO row can generate a new ODOMETRY position; matching
        is the higher-priority absolute correction, while VO supplies updates.
        """
        if row is None or row.get('status') != 'OK':
            return None
        try:
            segment = int(row['segment_id'])
        except (ValueError, KeyError, TypeError):
            return None
        if (self.segment_id != segment or self.origin is None
                or self.offset is None):
            return None
        e = self._finite(row.get('position_east_m'))
        n = self._finite(row.get('position_north_m'))
        t = self._finite(row.get('t_sim'))
        if e is None or n is None or t is None:
            return None

        # Correct towards matcher at bounded speed; keeps the EKF input
        # continuous even when a delayed fix disagrees with VO.
        residual = 0.0
        if self.target_offset is not None:
            de = self.target_offset[0] - self.offset[0]
            dn = self.target_offset[1] - self.offset[1]
            residual = math.hypot(de, dn)
            if self._last_tick_t is not None and 0 < t - self._last_tick_t < 2.0:
                limit = self.correction_mps * (t - self._last_tick_t)
                if residual <= limit:
                    self.offset = self.target_offset
                elif residual > 0:
                    f = limit / residual
                    self.offset = (self.offset[0] + de*f,
                                   self.offset[1] + dn*f)
            # On the very first step with a GPS1 anchor, do not jump.
        self._last_tick_t = t

        abs_e, abs_n = self.offset[0] + e, self.offset[1] + n
        if not all(math.isfinite(v) for v in (abs_e, abs_n)):
            return None
        local_e = abs_e - self.origin[0]
        local_n = abs_n - self.origin[1]
        used_match = self._new_match or (
            self.last_match_t is not None
            and 0 <= t - self.last_match_t <= self.recent_match_s
        )
        self._new_match = False
        return (local_n, local_e, 'matcher' if used_match else 'vo', residual)
