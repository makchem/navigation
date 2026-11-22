"""SITL-only ExternalNav ODOMETRY encoder for existing VO diagnostic rows.

No map truth, GPS positions or matcher fixes enter this adapter.  It deliberately
uses relative local NED coordinates: arbitrary first-frame origin, X north,
Y east, Z down.  The EKF handles alignment while GPS is the primary source.

NOT a flight-ready visual-inertial navigation estimator: yaw is AHRS-provided,
height/barometer is handled by ArduPilot, and sigma is uncalibrated.
"""
from __future__ import annotations

import math
import time


class ExternalNavPublisher:
    """Produce MAVLink ODOMETRY payloads only on fresh valid VO updates."""

    def __init__(self, *, min_good_frames: int = 12,
                 max_sigma_m: float = 30.0,
                 max_speed_mps: float = 65.0):
        self.min_good_frames = min_good_frames
        self.max_sigma_m = max_sigma_m
        self.max_speed_mps = max_speed_mps
        self._segment = None
        self._good = 0
        self.reset_counter = 0
        self._last_t = None

    def reset(self) -> None:
        """Tell EKF the local VO origin/track was reset."""
        self._segment = None
        self._good = 0
        self._last_t = None
        self.reset_counter = (self.reset_counter + 1) % 256

    @staticmethod
    def _finite(value):
        try:
            v = float(value)
        except (ValueError, TypeError):
            return None
        return v if math.isfinite(v) else None

    @staticmethod
    def _quaternion(roll: float, pitch: float, yaw: float) -> tuple[float, ...]:
        """NED to body attitude q (body -> local NED, Hamilton wxyz)."""
        cr, sr = math.cos(roll/2), math.sin(roll/2)
        cp, sp = math.cos(pitch/2), math.sin(pitch/2)
        cy, sy = math.cos(yaw/2), math.sin(yaw/2)
        return (
            cr*cp*cy + sr*sp*sy,
            sr*cp*cy - cr*sp*sy,
            cr*sp*cy + sr*cp*sy,
            cr*cp*sy - sr*sp*cy,
        )

    @staticmethod
    def _ned_to_body(n: float, e: float, d: float, q: tuple[float, ...]):
        """Transpose of body->NED rotation from quaternion."""
        w, x, y, z = q
        return (
            (1 - 2*(y*y + z*z))*n + (2*(x*y + w*z))*e + (2*(x*z - w*y))*d,
            (2*(x*y - w*z))*n + (1 - 2*(x*x + z*z))*e + (2*(y*z + w*x))*d,
            (2*(x*z + w*y))*n + (2*(y*z - w*x))*e + (1 - 2*(x*x + y*y))*d,
        )

    def payload(self, row: dict | None, attitude) -> dict | None:
        if row is None or attitude is None:
            return None
        try:
            segment = int(row['segment_id'])
        except (ValueError, KeyError, TypeError):
            return None

        if self._segment != segment:
            if self._segment is not None:
                self.reset_counter = (self.reset_counter + 1) % 256
            self._segment = segment
            self._good = 0
            self._last_t = None

        if row.get('status') != 'OK' or row.get('vo_ok') not in (True, 'True', 'true', 1):
            self._good = 0
            return None

        keys = ('position_east_m', 'position_north_m', 've_mps', 'vn_mps',
                'sigma_m', 't_sim', 'inlier_ratio')
        values = {key: self._finite(row.get(key)) for key in keys}
        angles = {key: self._finite(getattr(attitude, key, None))
                  for key in ('roll', 'pitch', 'yaw')}
        if None in values.values() or None in angles.values():
            self._good = 0
            return None

        e, n = values['position_east_m'], values['position_north_m']
        ve, vn = values['ve_mps'], values['vn_mps']
        sigma = values['sigma_m']
        t = values['t_sim']
        ratio = values['inlier_ratio']
        if sigma < 0 or sigma > self.max_sigma_m or math.hypot(ve, vn) > self.max_speed_mps:
            self._good = 0
            return None
        if self._last_t is not None and not 0 < t - self._last_t <= 0.5:
            self._good = 0
            self._last_t = t
            return None
        self._last_t = t
        self._good += 1
        if self._good < self.min_good_frames:
            return None

        q = self._quaternion(angles['roll'], angles['pitch'], angles['yaw'])
        vx, vy, vz = self._ned_to_body(vn, ve, 0.0, q)

        # Conservative placeholders; must be calibrated on independent truth.
        # ArduPilot reads sqrt(cov[0] + cov[6] + cov[11]) as pose error.
        horizontal_std = max(5.0, 3.0 * sigma)
        cov = [0.0] * 21
        cov[0] = cov[6] = horizontal_std**2
        cov[11] = 10.0**2  # ignored when POSZ is barometer
        cov[15] = cov[18] = math.radians(10.0)**2
        cov[20] = math.radians(20.0)**2

        return dict(
            time_usec=time.monotonic_ns() // 1000,
            frame_id=20,          # MAV_FRAME_LOCAL_FRD (north/east/down)
            child_frame_id=12,    # MAV_FRAME_BODY_FRD
            x=n, y=e, z=0.0,     # z is unused when EK3_SRC2_POSZ=1
            q=list(q),
            vx=vx, vy=vy, vz=vz,
            rollspeed=0.0, pitchspeed=0.0, yawspeed=0.0,
            pose_covariance=cov,
            velocity_covariance=[float('nan')]*21,
            reset_counter=self.reset_counter,
            estimator_type=0,    # not used by ArduPilot
            quality=max(1, min(100, int(round(100*max(0.0, min(1.0, ratio)))))),
        )
