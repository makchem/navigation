"""Tests for read-only VO CSV integration, independent of MAVLink/SITL."""

import csv

import cv2
import numpy as np

from navmod.georef import MapGeoref
from navmod.types import CameraFrame
from navmod.vo_diagnostics import VODiagnosticRunner


def textured_frames():
    rng = np.random.default_rng(42)
    texture = cv2.GaussianBlur(
        rng.integers(0, 255, (512, 512), dtype=np.uint8), (5, 5), 0
    )
    for i, t in enumerate((1.0, 1.2, 2.0, 2.2, 2.4)):
        matrix = np.float32([[1, 0, 6*i], [0, 1, 4*i]])
        yield CameraFrame(
            image=cv2.warpAffine(texture, matrix, (512, 512)),
            timestamp_s=t, height_agl_m=100.0, frame_id=i,
        )


def test_vo_diagnostics_segments_and_csv(tmp_path):
    geo = MapGeoref(51.08, 13.69, 51.06, 13.75, 8192, 5097)
    path = tmp_path / "run_vo.csv"
    with VODiagnosticRunner(path, geo, 60.0) as diag:
        frames = list(textured_frames())
        rows = [diag.observe(f, 0.0, 51.07, 13.72) for f in frames]

    assert [r["status"] for r in rows] == ["INIT", "OK", "LOST", "INIT", "OK"]
    assert [r["segment_id"] for r in rows] == [0, 0, 0, 1, 1]
    assert rows[1]["vo_ok"] and rows[4]["vo_ok"]
    assert rows[1]["sigma_m"] > 0
    assert rows[2]["position_error_m"] == ""

    with path.open(newline="") as file:
        saved = list(csv.DictReader(file))
    assert len(saved) == 5
    assert saved[2]["status"] == "LOST"
    assert saved[4]["status"] == "OK"


def test_vo_diagnostics_external_gap_restarts_segment(tmp_path):
    geo = MapGeoref(51.08, 13.69, 51.06, 13.75, 8192, 5097)
    with VODiagnosticRunner(tmp_path / "vo.csv", geo, 60.0) as diag:
        frames = list(textured_frames())
        assert diag.observe(frames[0], 0.0, 51.07, 13.72)["status"] == "INIT"
        diag.mark_gap()
        next_row = diag.observe(frames[1], 0.0, 51.07, 13.72)
        assert next_row["segment_id"] == 1
        assert next_row["status"] == "INIT"
