"""Tests for the frame augmentation pipeline."""

import cv2
import numpy as np
import pytest

from navmod.augment import (
    AugmentConfig,
    augment_frame,
    _apply_color_shift,
    _apply_brightness_contrast,
    _apply_gamma,
    _apply_noise,
    _apply_vignette,
    _apply_haze,
    _apply_jpeg,
)


@pytest.fixture
def sample_img():
    rng = np.random.RandomState(0)
    return rng.randint(50, 200, (256, 256, 3), dtype=np.uint8)


class TestIndividualTransforms:
    def test_color_shift_changes_pixels(self, sample_img):
        out = _apply_color_shift(sample_img, (10, -10, 5))
        assert not np.array_equal(out, sample_img)
        assert out.dtype == np.uint8

    def test_brightness_contrast(self, sample_img):
        out = _apply_brightness_contrast(sample_img, 20, 1.2)
        assert out.mean() > sample_img.mean()

    def test_gamma_brightens(self, sample_img):
        out = _apply_gamma(sample_img, 1.5)
        assert out.mean() > sample_img.mean()

    def test_gamma_identity(self, sample_img):
        out = _apply_gamma(sample_img, 1.0)
        np.testing.assert_array_equal(out, sample_img)

    def test_noise_adds_variance(self, sample_img):
        rng = np.random.RandomState(0)
        out = _apply_noise(sample_img, 20, rng)
        diff = np.abs(out.astype(float) - sample_img.astype(float))
        assert diff.mean() > 5

    def test_vignette_darkens_corners(self, sample_img):
        out = _apply_vignette(sample_img, 0.5)
        assert out[0, 0].mean() < sample_img[0, 0].mean()
        centre = sample_img.shape[0] // 2
        assert out[centre, centre].mean() >= sample_img[centre, centre].mean() * 0.9

    def test_haze_shifts_toward_haze_color(self, sample_img):
        out = _apply_haze(sample_img, 0.3, (255, 255, 255))
        assert out.mean() > sample_img.mean()

    def test_jpeg_changes_pixels(self, sample_img):
        out = _apply_jpeg(sample_img, 50)
        assert not np.array_equal(out, sample_img)

    def test_jpeg_100_is_noop(self, sample_img):
        out = _apply_jpeg(sample_img, 100)
        np.testing.assert_array_equal(out, sample_img)


class TestAugmentFrame:
    def test_default_config_changes_image(self, sample_img):
        out = augment_frame(sample_img)
        assert not np.array_equal(out, sample_img)
        assert out.shape[:2] == sample_img.shape[:2]
        assert out.dtype == np.uint8

    def test_seed_reproducibility(self, sample_img):
        cfg = AugmentConfig(seed=99)
        a = augment_frame(sample_img, cfg)
        b = augment_frame(sample_img, cfg)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self, sample_img):
        a = augment_frame(sample_img, AugmentConfig(seed=1))
        b = augment_frame(sample_img, AugmentConfig(seed=2))
        assert not np.array_equal(a, b)

    def test_grayscale_input(self):
        gray = np.full((100, 100), 128, dtype=np.uint8)
        out = augment_frame(gray, AugmentConfig(seed=0))
        assert out.ndim == 3
        assert out.shape[2] == 3

    def test_output_range(self, sample_img):
        cfg = AugmentConfig(brightness=50, contrast=1.5, noise_sigma=30, seed=0)
        out = augment_frame(sample_img, cfg)
        assert out.min() >= 0
        assert out.max() <= 255
