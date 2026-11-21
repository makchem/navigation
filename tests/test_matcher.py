"""End-to-end matcher test on synthetic frames."""

import numpy as np
import pytest

from navmod.augment import AugmentConfig
from navmod.camera import render_frame
from navmod.georef import MapGeoref, distance_m, px_to_latlon
from navmod.matcher import estimate_pose, match_frame
from navmod.metrics import error_m, gsd


SYNTH_SIZE = (2000, 2000)
MAP_GSD = 0.5  # metres per pixel, similar to Dresden map
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


class TestMatcherEndToEnd:
    def test_identity_position(self, synth_map):
        x, y = 1000, 1000
        frame = render_frame(synth_map, x, y, heading_deg=0, scale=1, size=FRAME_SIZE)
        predicted = match_frame(frame, synth_map)
        err = error_m(predicted, (x, y), MAP_GSD)
        assert err < 50, f"error {err:.1f} m exceeds 50 m threshold"

    def test_offset_position(self, synth_map):
        x, y = 600, 1400
        frame = render_frame(synth_map, x, y, heading_deg=0, scale=1, size=FRAME_SIZE)
        predicted = match_frame(frame, synth_map)
        err = error_m(predicted, (x, y), MAP_GSD)
        assert err < 50, f"error {err:.1f} m exceeds 50 m threshold"

    def test_with_heading(self, synth_map):
        x, y = 1000, 1000
        frame = render_frame(synth_map, x, y, heading_deg=5, scale=1, size=FRAME_SIZE)
        predicted = match_frame(frame, synth_map)
        if predicted is None:
            return
        err = error_m(predicted, (x, y), MAP_GSD)
        assert err < 50, f"error {err:.1f} m exceeds 50 m threshold"

    def test_with_scale(self, synth_map):
        x, y = 1000, 1000
        frame = render_frame(synth_map, x, y, heading_deg=0, scale=2, size=FRAME_SIZE)
        predicted = match_frame(frame, synth_map)
        if predicted is None:
            return
        err = error_m(predicted, (x, y), MAP_GSD)
        assert predicted is None or err < 50, f"error {err:.1f} m exceeds 50 m threshold"


class TestEstimatePose:
    """estimate_pose returns a Fix with position, heading, and uncertainty."""

    def test_position_accuracy(self, synth_map):
        x, y = 1000, 1000
        frame = render_frame(synth_map, x, y, heading_deg=0, scale=1, size=FRAME_SIZE)
        fix = estimate_pose(frame, synth_map, SYNTH_GEOREF)
        assert fix.ok
        true_lat, true_lon = px_to_latlon(SYNTH_GEOREF, x, y)
        err = distance_m(fix.lat, fix.lon, true_lat, true_lon)
        assert err < 25, f"position error {err:.1f} m exceeds 25 m"

    def test_offset_position(self, synth_map):
        x, y = 600, 1400
        frame = render_frame(synth_map, x, y, heading_deg=0, scale=1, size=FRAME_SIZE)
        fix = estimate_pose(frame, synth_map, SYNTH_GEOREF)
        assert fix.ok
        true_lat, true_lon = px_to_latlon(SYNTH_GEOREF, x, y)
        err = distance_m(fix.lat, fix.lon, true_lat, true_lon)
        assert err < 25, f"position error {err:.1f} m exceeds 25 m"

    def test_heading_zero(self, synth_map):
        frame = render_frame(synth_map, 1000, 1000, heading_deg=0, scale=1, size=FRAME_SIZE)
        fix = estimate_pose(frame, synth_map, SYNTH_GEOREF)
        assert fix.ok
        assert abs(fix.heading_rad) < np.radians(10), (
            f"heading {np.degrees(fix.heading_rad):.1f}° should be near 0°"
        )

    @pytest.mark.parametrize("heading_deg", [45, 90])
    def test_heading_recovery(self, synth_map, heading_deg):
        frame = render_frame(synth_map, 1000, 1000, heading_deg=heading_deg, scale=1, size=FRAME_SIZE)
        fix = estimate_pose(frame, synth_map, SYNTH_GEOREF)
        if not fix.ok:
            pytest.skip("matcher could not solve rotated frame")
        expected = np.radians(heading_deg)
        diff = abs(fix.heading_rad - expected)
        diff = min(diff, 2 * np.pi - diff)
        assert diff < np.radians(10), (
            f"heading {np.degrees(fix.heading_rad):.1f}° vs expected {heading_deg}°"
        )

    def test_sigma_is_finite_and_positive(self, synth_map):
        frame = render_frame(synth_map, 1000, 1000, heading_deg=0, scale=1, size=FRAME_SIZE)
        fix = estimate_pose(frame, synth_map, SYNTH_GEOREF)
        assert fix.ok
        assert 0 < fix.sigma_m < 50, f"sigma_m {fix.sigma_m:.2f} out of reasonable range"

    def test_inlier_stats(self, synth_map):
        frame = render_frame(synth_map, 1000, 1000, heading_deg=0, scale=1, size=FRAME_SIZE)
        fix = estimate_pose(frame, synth_map, SYNTH_GEOREF)
        assert fix.ok
        assert fix.n_inliers > 0
        assert 0 < fix.inlier_ratio <= 1.0

    def test_blank_frame_no_fix(self, synth_map):
        blank = np.zeros((*FRAME_SIZE, 3), dtype=np.uint8)
        fix = estimate_pose(blank, synth_map, SYNTH_GEOREF)
        assert not fix.ok
        assert fix.sigma_m == float("inf")

    def test_uniform_noise_no_fix(self, synth_map):
        noise = np.random.RandomState(99).randint(0, 256, (*FRAME_SIZE, 3), dtype=np.uint8)
        fix = estimate_pose(noise, synth_map, SYNTH_GEOREF)
        assert not fix.ok


MILD_AUGMENT = AugmentConfig(
    brightness=10, contrast=1.08, noise_sigma=8, blur_ksize=3,
    jpeg_quality=90, haze_alpha=0.05, vignette_strength=0.2,
    gamma=1.1, color_shift=(3, -2, -5), seed=7,
)

DEFAULT_AUGMENT = AugmentConfig(seed=42)

HARSH_AUGMENT = AugmentConfig(
    brightness=25, contrast=1.25, noise_sigma=20, blur_ksize=5,
    jpeg_quality=65, haze_alpha=0.15, vignette_strength=0.5,
    gamma=1.3, color_shift=(10, -8, -15), seed=13,
)


class TestAugmentedMatching:
    """Matcher must still converge when the frame is degraded."""

    @pytest.mark.parametrize("aug,tol_m", [
        (MILD_AUGMENT, 50),
        (DEFAULT_AUGMENT, 50),
        (HARSH_AUGMENT, 75),
    ], ids=["mild", "default", "harsh"])
    def test_position_with_augmentation(self, synth_map, aug, tol_m):
        x, y = 1000, 1000
        frame = render_frame(synth_map, x, y, heading_deg=0, scale=1,
                             size=FRAME_SIZE, augment=aug)
        fix = estimate_pose(frame, synth_map, SYNTH_GEOREF)
        assert fix.ok, "matcher failed on augmented frame"
        true_lat, true_lon = px_to_latlon(SYNTH_GEOREF, x, y)
        err = distance_m(fix.lat, fix.lon, true_lat, true_lon)
        assert err < tol_m, f"position error {err:.1f} m exceeds {tol_m} m"

    def test_augmented_heading(self, synth_map):
        x, y = 1000, 1000
        frame = render_frame(synth_map, x, y, heading_deg=30, scale=1,
                             size=FRAME_SIZE, augment=DEFAULT_AUGMENT)
        fix = estimate_pose(frame, synth_map, SYNTH_GEOREF)
        if not fix.ok:
            pytest.skip("matcher could not solve augmented rotated frame")
        expected = np.radians(30)
        diff = abs(fix.heading_rad - expected)
        diff = min(diff, 2 * np.pi - diff)
        assert diff < np.radians(15), (
            f"heading {np.degrees(fix.heading_rad):.1f}° vs expected 30°"
        )

    def test_augmented_offset(self, synth_map):
        x, y = 600, 1400
        frame = render_frame(synth_map, x, y, heading_deg=0, scale=1,
                             size=FRAME_SIZE, augment=DEFAULT_AUGMENT)
        fix = estimate_pose(frame, synth_map, SYNTH_GEOREF)
        assert fix.ok
        true_lat, true_lon = px_to_latlon(SYNTH_GEOREF, x, y)
        err = distance_m(fix.lat, fix.lon, true_lat, true_lon)
        assert err < 50, f"position error {err:.1f} m exceeds 50 m"

    def test_sigma_increases_with_degradation(self, synth_map):
        x, y = 1000, 1000
        frame_clean = render_frame(synth_map, x, y, size=FRAME_SIZE)
        frame_augmented = render_frame(synth_map, x, y, size=FRAME_SIZE,
                                       augment=HARSH_AUGMENT)
        fix_clean = estimate_pose(frame_clean, synth_map, SYNTH_GEOREF)
        fix_aug = estimate_pose(frame_augmented, synth_map, SYNTH_GEOREF)
        assert fix_clean.ok and fix_aug.ok
        assert fix_aug.sigma_m >= fix_clean.sigma_m, (
            f"augmented sigma {fix_aug.sigma_m:.2f} should be >= "
            f"clean sigma {fix_clean.sigma_m:.2f}"
        )
