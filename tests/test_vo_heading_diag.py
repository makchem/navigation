"""Heading diagnostics must not alter VO input or status."""
import math
import cv2
import numpy as np
import pytest
from navmod.georef import MapGeoref
from navmod.types import CameraFrame
from navmod.vo_diagnostics import VODiagnosticRunner


def test_heading_and_wall_time_logging(tmp_path):
    image = cv2.GaussianBlur(np.random.default_rng(42).integers(
        0, 255, (512, 512), dtype=np.uint8), (5, 5), 0)
    georef = MapGeoref(51.08, 13.69, 51.06, 13.75, 8192, 5097)
    with VODiagnosticRunner(tmp_path / 'diag.csv', georef, 60.) as diag:
        # Angles straddle +/-180; shortest difference is +2 degrees.
        f1 = CameraFrame(image, 1.0, 100., 0)
        f2 = CameraFrame(image, 1.2, 100., 1)
        one = diag.observe(f1, math.radians(-179), 51.07, 13.72,
                           gt_yaw_rad=math.radians(179))
        two = diag.observe(f2, 0., 51.07, 13.72,
                           gt_yaw_rad=math.radians(-5))
    assert one['yaw_diff_deg'] == pytest.approx(2., abs=1e-6)
    assert two['yaw_diff_deg'] == pytest.approx(5., abs=1e-6)
    assert one['wall_dt_s'] == ''
    assert two['wall_dt_s'] >= 0
    assert one['status'] == 'INIT'
