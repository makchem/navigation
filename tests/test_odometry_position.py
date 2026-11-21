"""Position accumulation checks for the experimental VisualOdometry extension."""

import math

import cv2
import numpy as np

from navmod.odometry import VisualOdometry, camera_gsd
from navmod.types import CameraFrame


def _texture():
    rng = np.random.default_rng(42)
    raw = rng.integers(0, 255, (512, 512), dtype=np.uint8)
    return cv2.GaussianBlur(raw, (5, 5), 0)


def _frame(texture, x, y, timestamp, frame_id):
    matrix = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(texture, matrix, (512, 512))
    return CameraFrame(
        image=shifted,
        timestamp_s=timestamp,
        height_agl_m=100.0,
        frame_id=frame_id,
    )


def test_position_and_sigma_accumulate():
    texture = _texture()
    vo = VisualOdometry(fov_diag_deg=60)
    origin = vo.update(_frame(texture, 0, 0, 1.0, 0))
    step1 = vo.update(_frame(texture, 6, 4, 1.2, 1))
    step2 = vo.update(_frame(texture, 12, 8, 1.4, 2))

    gsd = camera_gsd(100, 60, 512, 512)
    assert not origin.ok
    assert step1.ok and step2.ok
    assert abs(step2.position_east_m + 12 * gsd) < gsd
    assert abs(step2.position_north_m - 8 * gsd) < gsd
    assert 0 < step1.sigma_m < step2.sigma_m
    assert step2.sigma_delta_m > 0


def test_tracking_loss_requires_reset():
    texture = _texture()
    vo = VisualOdometry(fov_diag_deg=60)
    vo.update(_frame(texture, 0, 0, 1.0, 0))
    assert vo.update(_frame(texture, 6, 4, 1.2, 1)).ok

    gap = vo.update(_frame(texture, 12, 8, 2.0, 2))
    next_flow = vo.update(_frame(texture, 18, 12, 2.2, 3))
    assert not gap.ok
    assert not next_flow.ok
    assert math.isinf(gap.sigma_m)
    assert math.isinf(next_flow.sigma_m)
    assert math.isnan(next_flow.position_east_m)

    vo.reset()  # Start a new relative trajectory (NOT an absolute fix).
    assert not vo.update(_frame(texture, 12, 8, 3.0, 4)).ok
    assert vo.update(_frame(texture, 18, 12, 3.2, 5)).ok
