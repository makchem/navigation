"""Synthetic camera geometry regression tests: movement, yaw, zoom.

The renderer is the actual camera-model API in src/navmod/camera.py.
Positions are *map pixels*, not flight-state inputs to the estimator.
"""
import math
import cv2
import numpy as np
import pytest
from navmod.camera import render_frame
from navmod.odometry import VisualOdometry, camera_gsd
from navmod.types import CameraFrame

WIDTH, HEIGHT = 512, 512
FOV_DIAG = 60.0
H0 = 100.0
MAP_GSD = camera_gsd(H0, FOV_DIAG, WIDTH, HEIGHT)

@pytest.fixture(scope="module")
def terrain():
    rng = np.random.default_rng(20261121)
    image = rng.integers(0, 256, (1536, 1536), dtype=np.uint8)
    return cv2.GaussianBlur(image, (5, 5), 0)

def frame(terrain, *, x=768, y=768, yaw_deg=0.0, h_m=H0, t=1., idx=0):
    image = render_frame(
        terrain, x=x, y=y, heading_deg=yaw_deg,
        scale=camera_gsd(h_m, FOV_DIAG, WIDTH, HEIGHT)/MAP_GSD,
        size=(WIDTH, HEIGHT), augment=False,
    )
    return CameraFrame(image=image, timestamp_s=t, height_agl_m=h_m, frame_id=idx)

def _run(terrain, first, second):
    vo=VisualOdometry(fov_diag_deg=FOV_DIAG)
    vo.update(frame(terrain,**first))
    return vo.update(frame(terrain,**second),yaw_rad=math.radians(first.get('yaw_deg',0.0)))

def test_pure_yaw_is_not_translation(terrain):
    """Rotation around camera centre must not create false movement."""
    r=_run(terrain,
       dict(yaw_deg=15,t=1.,idx=0),
       dict(yaw_deg=21,t=1.2,idx=1))
    assert r.ok,(r.n_inliers,r.inlier_ratio,r.rms_px)
    assert abs(r.delta_east_m)<0.3,r.delta_east_m
    assert abs(r.delta_north_m)<0.3,r.delta_north_m

def test_yaw_and_translation_together(terrain):
    r=_run(terrain,
       dict(x=768,y=768,yaw_deg=15,t=1.,idx=0),
       dict(x=783,y=761,yaw_deg=19,t=1.2,idx=1))
    assert r.ok,(r.n_inliers,r.inlier_ratio,r.rms_px)
    assert r.delta_east_m==pytest.approx(15*MAP_GSD,abs=0.3)
    assert r.delta_north_m==pytest.approx(7*MAP_GSD,abs=0.3)

def test_zoom_and_translation(terrain):
    """Image pixel-to-metre mapping uses FIRST frame's camera scale."""
    r=_run(terrain,
       dict(x=768,y=768,h_m=H0,t=1.,idx=0),
       dict(x=774,y=765,h_m=110.,t=1.2,idx=1))
    assert r.ok,(r.n_inliers,r.inlier_ratio,r.rms_px)
    assert r.delta_east_m==pytest.approx(6*MAP_GSD,abs=0.02)
    assert r.delta_north_m==pytest.approx(3*MAP_GSD,abs=0.02)


def test_yaw_wraparound(terrain):
    """Heading wrap from +179 to -179 degrees is a small rotation."""
    r = _run(terrain,
        dict(x=768, y=768, yaw_deg=179, t=1., idx=0),
        dict(x=774, y=765, yaw_deg=-179, t=1.2, idx=1),
    )
    assert r.ok, (r.n_inliers, r.inlier_ratio, r.rms_px)
    assert r.delta_east_m == pytest.approx(6 * MAP_GSD, abs=0.1)
    assert r.delta_north_m == pytest.approx(3 * MAP_GSD, abs=0.1)
