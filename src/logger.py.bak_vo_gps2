"""One CSV per SITL run; all timestamps in the CSV are autopilot boot seconds.

This module deliberately has no dependency on MAVLink or OpenCV. Each call to
log_tick represents one processed navigation tick (not an EKF fusion event).
"""

from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


class RunLogger:
    FIELDS = (
        "t_sim", "frame_id", "gt_lat", "gt_lon", "gt_heading", "gt_alt",
        "fix_lat", "fix_lon", "fix_heading", "sigma_m", "ok",
        "n_inliers", "n_total", "inlier_ratio", "branch_won",
        "sent_to_ekf", "gps_state",
    )
    BRANCH_FIELDS = (
        "t_sim", "frame_id", "branch", "n_keypoints", "n_matches",
        "n_inliers", "rms_px", "time_ms",
    )
    GPS_STATES = frozenset({"ok", "jammed", "spoofed", "off"})
    BRANCHES = frozenset({"", "texture", "structure", "fused"})

    def __init__(self, directory: str | Path = "logs/", flush_every: int = 10):
        if flush_every <= 0:
            raise ValueError("flush_every must be positive")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.flush_every = flush_every
        self._stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        self._file = None
        self._branch_file = None
        self._branch_writer = None
        self._branch_pending = 0
        self._pending = 0
        self._closed = False
        self._last_frame_id = -1

        # Exclusive creation ensures no run overwrites another run started
        # within the same minute. Typical filename uses the exact spec.
        for i in range(1000):
            tail = "" if i == 0 else f"_{i:02d}"
            self.path = self.directory / f"run_{self._stamp}{tail}.csv"
            try:
                self._file = self.path.open("x", newline="", encoding="utf-8")
                break
            except FileExistsError:
                continue
        else:
            raise RuntimeError("Could not allocate unique run CSV filename")

        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        self._writer.writeheader()

    @staticmethod
    def _value(obj: Any, *names: str, default: Any = "") -> Any:
        if obj is None:
            return default
        for name in names:
            if isinstance(obj, Mapping) and name in obj:
                return obj[name]
            if hasattr(obj, name):
                return getattr(obj, name)
        return default

    @staticmethod
    def _finite_or_blank(value: Any) -> Any:
        if value is None:
            return ""
        try:
            return value if math.isfinite(float(value)) else ""
        except (ValueError, TypeError):
            return ""

    def log_tick(
        self, *, t_sim: float, frame_id: int, gt: Any, fix: Any,
        branch_won: str = "", sent_to_ekf: bool = False,
        gps_state: str = "off",
    ) -> None:
        if self._closed:
            raise RuntimeError("RunLogger is closed")
        if not math.isfinite(float(t_sim)) or t_sim < 0:
            raise ValueError("t_sim must be nonnegative boot seconds")
        if int(frame_id) != frame_id or frame_id <= self._last_frame_id:
            raise ValueError("frame_id must increase, starting at 0")
        if branch_won not in self.BRANCHES:
            raise ValueError(f"Invalid branch_won: {branch_won!r}")
        if gps_state not in self.GPS_STATES:
            raise ValueError(f"Invalid gps_state: {gps_state!r}")

        ok = bool(self._value(fix, "ok", default=False))
        row = {
            "t_sim": t_sim,
            "frame_id": frame_id,
            "gt_lat": self._finite_or_blank(self._value(gt, "lat", "gt_lat")),
            "gt_lon": self._finite_or_blank(self._value(gt, "lon", "gt_lon")),
            "gt_heading": self._finite_or_blank(self._value(gt, "yaw_rad", "heading_rad", "gt_heading")),
            "gt_alt": self._finite_or_blank(self._value(gt, "alt_m", "gt_alt")),
            "fix_lat": self._finite_or_blank(self._value(fix, "lat")) if ok else "",
            "fix_lon": self._finite_or_blank(self._value(fix, "lon")) if ok else "",
            "fix_heading": self._finite_or_blank(self._value(fix, "heading_rad")) if ok else "",
            "sigma_m": self._value(fix, "sigma_m", default=float("inf")),
            "ok": ok,
            "n_inliers": self._value(fix, "n_inliers", default=0),
            "n_total": self._value(fix, "n_total", default=0),
            "inlier_ratio": self._value(fix, "inlier_ratio", default=0.0),
            "branch_won": branch_won,
            "sent_to_ekf": bool(sent_to_ekf),
            "gps_state": gps_state,
        }
        self._writer.writerow(row)
        self._last_frame_id = frame_id
        self._pending += 1
        if self._pending >= self.flush_every:
            self._file.flush()
            self._pending = 0

    def log_branch(
        self, *, t_sim: float, frame_id: int, branch: str,
        n_keypoints: int, n_matches: int, n_inliers: int,
        rms_px: float | None, time_ms: float,
    ) -> None:
        """Optional details CSV. Called by future multi-branch matcher only."""
        if self._closed:
            raise RuntimeError("RunLogger is closed")
        if branch not in {"texture", "structure"}:
            raise ValueError("branch must be 'texture' or 'structure'")
        if self._branch_file is None:
            self.branch_path = self.path.with_name(self.path.stem + "_matcher.csv")
            self._branch_file = self.branch_path.open("x", newline="", encoding="utf-8")
            self._branch_writer = csv.DictWriter(self._branch_file, fieldnames=self.BRANCH_FIELDS)
            self._branch_writer.writeheader()
        self._branch_writer.writerow({
            "t_sim": t_sim, "frame_id": frame_id, "branch": branch,
            "n_keypoints": n_keypoints, "n_matches": n_matches,
            "n_inliers": n_inliers, "rms_px": self._finite_or_blank(rms_px),
            "time_ms": time_ms,
        })
        self._branch_pending += 1
        if self._branch_pending >= self.flush_every:
            self._branch_file.flush()
            self._branch_pending = 0

    def close(self) -> None:
        if not self._closed:
            self._file.flush()
            self._file.close()
            if self._branch_file is not None:
                self._branch_file.flush()
                self._branch_file.close()
            self._closed = True

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
