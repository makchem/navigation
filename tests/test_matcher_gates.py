"""Failure-path and gate tests for the matcher."""

import math

import numpy as np
import pytest

from navmod.camera import render_frame
from navmod.georef import MapGeoref, distance_m, px_to_latlon
from navmod.matcher import MapIndex, _no_fix, _sigma_from_lut, estimate_pose


SYNTH_SIZE = (2000, 2000)
FRAME_SIZE = (512, 512)

SYNTH_GEOREF = MapGeoref(
    top_left_lat=51.080,
    top_left_lon=13.691,
    bottom_right_lat=51.062,
    bottom_right_lon=13.715,
    image_width_px=SYNTH_SIZE[1],
    image_height_px=SYNTH_SIZE[0],
)


@pytest.fixture(scope="module")
def synth_map():
    import cv2

    rng = np.random.RandomState(42)
    img = np.full((*SYNTH_SIZE, 3), 200, dtype=np.uint8)
    for _ in range(300):
        x1, y1 = rng.randint(0, SYNTH_SIZE[1]), rng.randint(0, SYNTH_SIZE[0])
        w, h = rng.randint(10, 80), rng.randint(10, 80)
        color = tuple(int(c) for c in rng.randint(0, 180, 3))
        cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), color, -1)
    for _ in range(200):
        cx, cy = rng.randint(0, SYNTH_SIZE[1]), rng.randint(0, SYNTH_SIZE[0])
        r = int(rng.randint(5, 40))
        color = tuple(int(c) for c in rng.randint(0, 180, 3))
        cv2.circle(img, (cx, cy), r, color, -1)
    return img


class TestNoFix:
    def test_nan_coordinates(self):
        fix = _no_fix()
        assert math.isnan(fix.lat)
        assert math.isnan(fix.lon)
        assert math.isnan(fix.heading_rad)

    def test_infinite_sigma(self):
        fix = _no_fix()
        assert fix.sigma_m == float("inf")

    def test_not_ok(self):
        fix = _no_fix()
        assert not fix.ok

    def test_distinct_objects(self):
        a = _no_fix()
        b = _no_fix()
        assert a is not b


class TestSigmaFromLut:
    LUT = [
        {"max_ratio": 0.25, "ok": False},
        {"max_ratio": 0.35, "sigma_m": 1.0},
        {"max_ratio": 0.50, "sigma_m": 0.5},
        {"max_ratio": 1.00, "sigma_m": 0.2},
    ]

    def test_reject_branch_returns_inf(self):
        sigma, ok = _sigma_from_lut(0.20, self.LUT)
        assert not ok
        assert sigma == float("inf")

    def test_ok_branch_returns_finite_sigma(self):
        sigma, ok = _sigma_from_lut(0.30, self.LUT)
        assert ok
        assert sigma == 1.0


class TestScaleGate:
    def test_rejects_wrong_scale(self, synth_map):
        x, y = 1000, 1000
        frame = render_frame(synth_map, x, y, heading_deg=0, scale=1.0,
                             size=FRAME_SIZE)
        fix = estimate_pose(frame, synth_map, SYNTH_GEOREF, expected_scale=3.0)
        assert not fix.ok


class TestWindowedSearch:
    def test_windowed_agrees_with_global(self, synth_map):
        x, y = 1000, 1000
        frame = render_frame(synth_map, x, y, heading_deg=0, scale=1.0,
                             size=FRAME_SIZE)

        fix_global = estimate_pose(frame, synth_map, SYNTH_GEOREF)
        fix_window = estimate_pose(
            frame, synth_map, SYNTH_GEOREF,
            prior_px=(x, y), search_radius_px=750,
        )

        assert fix_global.ok, "global search must converge"
        assert fix_window.ok, "windowed search must converge"

        err_between = distance_m(
            fix_global.lat, fix_global.lon,
            fix_window.lat, fix_window.lon,
        )
        assert err_between < 5, (
            f"global vs windowed disagreement {err_between:.1f} m exceeds 5 m"
        )

        true_lat, true_lon = px_to_latlon(SYNTH_GEOREF, x, y)
        err_window = distance_m(fix_window.lat, fix_window.lon, true_lat, true_lon)
        assert err_window < 25, (
            f"windowed error {err_window:.1f} m exceeds 25 m"
        )
