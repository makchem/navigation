"""Read-only SITL diagnostic for relative visual odometry.

Uses ground truth ONLY for scoring. Ground-truth lat/lon never enters the
VisualOdometry estimator; it receives only image, height and attitude yaw.
No MAVLink messages are sent by this module.
"""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path

from navmod.georef import MapGeoref, latlon_to_enu
from navmod.odometry import VisualOdometry
from navmod.types import CameraFrame


class VODiagnosticRunner:
    FIELDS = (
        "t_sim", "frame_id", "segment_id", "status", "vo_ok", "dt_s",
        "dx_px", "dy_px", "delta_east_m", "delta_north_m",
        "position_east_m", "position_north_m", "ve_mps", "vn_mps",
        "sigma_delta_m", "sigma_m", "n_tracked", "n_inliers",
        "inlier_ratio", "rms_px", "gt_east_m", "gt_north_m",
        "position_error_m", "vo_time_ms", "simstate_age_ms",
        "attitude_age_ms",
    )

    def __init__(
        self, path: str | Path, georef: MapGeoref, fov_diag_deg: float,
        flush_every: int = 10,
    ):
        if flush_every < 1:
            raise ValueError("flush_every must be positive")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.georef = georef
        self.vo = VisualOdometry(fov_diag_deg=fov_diag_deg)
        if not hasattr(self.vo, "reset"):
            raise RuntimeError("Install the VO position/sigma patch before running diagnostics")
        self._file = self.path.open("x", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        self._writer.writeheader()
        self._pending = 0
        self._flush_every = flush_every
        self._segment_id = 0
        self._needs_reset = False
        self._gt_origin: tuple[float, float] | None = None
        self._previous_yaw: float | None = None

    def mark_gap(self) -> None:
        """Next frame must start a new segment (link loss / missing attitude)."""
        self._needs_reset = True

    @staticmethod
    def _finite(value: float) -> float | str:
        return float(value) if math.isfinite(value) else ""

    def observe(
        self, frame: CameraFrame, yaw_rad: float, gt_lat: float,
        gt_lon: float, simstate_age_ms: float = float("nan"),
        attitude_age_ms: float = float("nan"),
    ) -> dict:
        """Process a unique camera frame; truth is used ONLY for error metrics."""
        if not all(math.isfinite(x) for x in (yaw_rad, gt_lat, gt_lon)):
            raise ValueError("Expected finite attitude yaw and SITL truth for diagnostics")

        if self._needs_reset:
            self.vo.reset()
            self._segment_id += 1
            self._gt_origin = None
            self._previous_yaw = None
            self._needs_reset = False

        truth = latlon_to_enu(self.georef, gt_lat, gt_lon)
        if self._gt_origin is None:
            self._gt_origin = truth
        truth_rel = (
            truth[0] - self._gt_origin[0],
            truth[1] - self._gt_origin[1],
        )

        had_previous = self.vo.prev_gray is not None
        # This is the previous frame's AHRS yaw, not SIMSTATE ground-truth yaw.
        yaw_for_step = self._previous_yaw if self._previous_yaw is not None else yaw_rad
        start = time.perf_counter()
        result = self.vo.update(frame, yaw_rad=yaw_for_step)
        duration_ms = (time.perf_counter() - start) * 1000.0
        self._previous_yaw = yaw_rad

        if not had_previous:
            status = "INIT"
        elif result.ok:
            status = "OK"
        else:
            status = "LOST"
            self.mark_gap()  # Next available frame becomes a new origin.

        err = (
            math.hypot(
                result.position_east_m - truth_rel[0],
                result.position_north_m - truth_rel[1],
            ) if result.ok else float("nan")
        )
        row = {
            "t_sim": frame.timestamp_s,
            "frame_id": frame.frame_id,
            "segment_id": self._segment_id,
            "status": status,
            "vo_ok": result.ok,
            "dt_s": self._finite(result.dt_s),
            "dx_px": self._finite(result.dx_px),
            "dy_px": self._finite(result.dy_px),
            "delta_east_m": self._finite(result.delta_east_m),
            "delta_north_m": self._finite(result.delta_north_m),
            "position_east_m": self._finite(result.position_east_m),
            "position_north_m": self._finite(result.position_north_m),
            "ve_mps": self._finite(result.ve_mps),
            "vn_mps": self._finite(result.vn_mps),
            "sigma_delta_m": self._finite(result.sigma_delta_m),
            "sigma_m": self._finite(result.sigma_m),
            "n_tracked": result.n_tracked,
            "n_inliers": result.n_inliers,
            "inlier_ratio": result.inlier_ratio,
            "rms_px": self._finite(result.rms_px),
            "gt_east_m": truth_rel[0],
            "gt_north_m": truth_rel[1],
            "position_error_m": self._finite(err),
            "vo_time_ms": duration_ms,
            "simstate_age_ms": self._finite(simstate_age_ms),
            "attitude_age_ms": self._finite(attitude_age_ms),
        }
        self._writer.writerow(row)
        self._pending += 1
        if self._pending >= self._flush_every:
            self._file.flush()
            self._pending = 0
        return row

    def close(self) -> None:
        if not self._file.closed:
            self._file.flush()
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
