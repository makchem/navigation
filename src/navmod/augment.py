"""Realistic degradations for synthetic drone frames.

Applies sensor noise, lighting variation, compression artifacts, and
atmospheric effects so that rendered frames are visually distinct from
the reference map — matching them tests the matcher for real.

The same pipeline can preprocess map tiles into "drone-like" textures
for SITL visual simulation.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class AugmentConfig:
    """Tuneable augmentation parameters.

    Defaults model a typical consumer drone camera at ~100-200 m AGL in
    daylight.  Set ``seed`` for reproducible frames in tests.
    """
    brightness: float = 15.0
    contrast: float = 1.12
    noise_sigma: float = 12.0
    blur_ksize: int = 3
    jpeg_quality: int = 82
    haze_alpha: float = 0.08
    haze_color: tuple[int, int, int] = (210, 220, 230)
    color_shift: tuple[int, int, int] = (5, -3, -8)
    vignette_strength: float = 0.35
    gamma: float = 1.15
    seed: int | None = None


# -- individual transforms ---------------------------------------------------

def _apply_color_shift(img: np.ndarray, shift: tuple[int, int, int]) -> np.ndarray:
    return np.clip(img.astype(np.int16) + np.array(shift, dtype=np.int16), 0, 255).astype(np.uint8)


def _apply_brightness_contrast(img: np.ndarray, brightness: float, contrast: float) -> np.ndarray:
    out = img.astype(np.float32) * contrast + brightness
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    if abs(gamma - 1.0) < 0.01:
        return img
    lut = np.clip((np.arange(256) / 255.0) ** (1.0 / gamma) * 255, 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def _apply_noise(img: np.ndarray, sigma: float, rng: np.random.RandomState) -> np.ndarray:
    noise = rng.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _apply_blur(img: np.ndarray, ksize: int) -> np.ndarray:
    if ksize < 3:
        return img
    k = ksize | 1
    return cv2.GaussianBlur(img, (k, k), 0)


def _apply_vignette(img: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return img
    h, w = img.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2, h / 2
    r_max = np.sqrt(cx ** 2 + cy ** 2)
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / r_max
    mask = 1.0 - strength * r ** 2
    mask = np.clip(mask, 0, 1)
    if img.ndim == 3:
        mask = mask[:, :, np.newaxis]
    return np.clip(img.astype(np.float32) * mask, 0, 255).astype(np.uint8)


def _apply_haze(img: np.ndarray, alpha: float, color: tuple[int, int, int]) -> np.ndarray:
    if alpha <= 0:
        return img
    haze = np.full_like(img, color, dtype=np.uint8)
    return cv2.addWeighted(img, 1 - alpha, haze, alpha, 0)


def _apply_jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    if quality >= 100:
        return img
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return img
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# -- composite ----------------------------------------------------------------

def augment_frame(img: np.ndarray, cfg: AugmentConfig | None = None) -> np.ndarray:
    """Apply the full degradation pipeline to *img* (BGR, uint8).

    Order matters: colour/lighting first, then spatial, then compression —
    mimicking the real signal chain (scene → optics → sensor → codec).
    """
    if cfg is None:
        cfg = AugmentConfig()

    rng = np.random.RandomState(cfg.seed)

    out = img.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    out = _apply_color_shift(out, cfg.color_shift)
    out = _apply_brightness_contrast(out, cfg.brightness, cfg.contrast)
    out = _apply_gamma(out, cfg.gamma)
    out = _apply_haze(out, cfg.haze_alpha, cfg.haze_color)
    out = _apply_vignette(out, cfg.vignette_strength)
    out = _apply_blur(out, cfg.blur_ksize)
    out = _apply_noise(out, cfg.noise_sigma, rng)
    out = _apply_jpeg(out, cfg.jpeg_quality)

    return out
