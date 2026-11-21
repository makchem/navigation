"""Realistic degradations for synthetic drone frames.

Applies sensor noise, lighting variation, compression artifacts, and
atmospheric effects so that rendered frames are visually distinct from
the reference map — matching them tests the matcher for real.

The same pipeline can preprocess map tiles into "drone-like" textures
for SITL visual simulation.
"""

from dataclasses import dataclass

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
    shift_px: float = 3.0
    rotation_deg: float = 2.0
    scale_jitter: float = 0.03
    perspective_px: float = 5.0
    motion_blur_px: float = 4.0
    seed: int | None = None


# -- geometric transforms ----------------------------------------------------

def _apply_shift(img: np.ndarray, max_px: float, rng: np.random.RandomState) -> np.ndarray:
    if max_px <= 0:
        return img
    dx = rng.uniform(-max_px, max_px)
    dy = rng.uniform(-max_px, max_px)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                          borderMode=cv2.BORDER_REFLECT_101)


def _apply_rotation(img: np.ndarray, max_deg: float, rng: np.random.RandomState) -> np.ndarray:
    if max_deg <= 0:
        return img
    angle = rng.uniform(-max_deg, max_deg)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def _apply_scale(img: np.ndarray, jitter: float, rng: np.random.RandomState) -> np.ndarray:
    if jitter <= 0:
        return img
    factor = rng.uniform(1.0 - jitter, 1.0 + jitter)
    h, w = img.shape[:2]
    new_h, new_w = int(round(h * factor)), int(round(w * factor))
    interp = cv2.INTER_LINEAR if factor >= 1.0 else cv2.INTER_AREA
    scaled = cv2.resize(img, (new_w, new_h), interpolation=interp)
    # center-crop or center-pad back to original size
    sh, sw = scaled.shape[:2]
    y0 = (sh - h) // 2
    x0 = (sw - w) // 2
    if factor >= 1.0:
        return scaled[y0:y0 + h, x0:x0 + w]
    # scaled smaller — pad with border reflection
    out = cv2.copyMakeBorder(scaled, -y0, h - sh - (-y0), -x0, w - sw - (-x0),
                             cv2.BORDER_REFLECT_101)
    return out[:h, :w]


def _apply_perspective(img: np.ndarray, max_px: float, rng: np.random.RandomState) -> np.ndarray:
    if max_px <= 0:
        return img
    h, w = img.shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    offsets = rng.uniform(-max_px, max_px, (4, 2)).astype(np.float32)
    dst = corners + offsets
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def _apply_motion_blur(img: np.ndarray, max_px: float, rng: np.random.RandomState) -> np.ndarray:
    if max_px < 1.0:
        return img
    length = rng.uniform(1.0, max_px)
    angle = rng.uniform(0, 360)
    ksize = int(np.ceil(length)) | 1
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    center = ksize // 2
    rad = np.radians(angle)
    dx, dy = np.cos(rad), np.sin(rad)
    for i in range(ksize):
        t = i - center
        x = int(round(center + t * dx))
        y = int(round(center + t * dy))
        if 0 <= x < ksize and 0 <= y < ksize:
            kernel[y, x] = 1.0
    kernel /= kernel.sum()
    return cv2.filter2D(img, -1, kernel)


# -- photometric transforms --------------------------------------------------

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

    # geometric (camera motion) — before sensor capture
    out = _apply_shift(out, cfg.shift_px, rng)
    out = _apply_rotation(out, cfg.rotation_deg, rng)
    out = _apply_scale(out, cfg.scale_jitter, rng)
    out = _apply_perspective(out, cfg.perspective_px, rng)
    out = _apply_motion_blur(out, cfg.motion_blur_px, rng)

    # photometric (sensor + codec) — per-frame randomization around config
    def _jitter(val: float) -> float:
        return val * rng.uniform(0.7, 1.3)

    brightness = _jitter(cfg.brightness)
    contrast = max(0.5, _jitter(cfg.contrast))
    noise_sigma = max(0.0, _jitter(cfg.noise_sigma))
    haze_alpha = min(0.5, max(0.0, _jitter(cfg.haze_alpha)))
    gamma = max(0.5, min(2.0, _jitter(cfg.gamma)))

    out = _apply_color_shift(out, cfg.color_shift)
    out = _apply_brightness_contrast(out, brightness, contrast)
    out = _apply_gamma(out, gamma)
    out = _apply_haze(out, haze_alpha, cfg.haze_color)
    out = _apply_vignette(out, cfg.vignette_strength)
    out = _apply_blur(out, cfg.blur_ksize)
    out = _apply_noise(out, noise_sigma, rng)
    out = _apply_jpeg(out, cfg.jpeg_quality)

    return out
