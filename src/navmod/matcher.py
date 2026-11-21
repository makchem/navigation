"""Frame-to-map matching and pose recovery.

Reminder from the roadmap: prefer cv2.estimateAffinePartial2D over
findHomography for nadir imagery. A downward-looking camera over flat terrain
is a 4-DOF similarity (rotation, scale, translation); fitting 8 DOF spends the
extra freedom on noise and can produce confident nonsense on thin inlier sets.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from navmod.georef import MapGeoref, px_to_latlon
from navmod.types import Fix

FRAME_NFEATURES = 1500

_map_cache: dict[int, "MapIndex"] = {}


def _no_fix() -> Fix:
    return Fix(
        lat=float("nan"), lon=float("nan"), heading_rad=float("nan"),
        sigma_m=float("inf"), ok=False,
    )


def _sigma_from_lut(inlier_ratio: float, lut: list[dict]) -> tuple[float, bool]:
    """Return (sigma_m, ok) from the lookup table."""
    for entry in lut:
        if inlier_ratio < entry["max_ratio"]:
            if not entry.get("ok", True):
                return float("inf"), False
            return entry["sigma_m"], True
    last = lut[-1]
    return last["sigma_m"], True


def _default_nfeatures(img):
    """Scale ORB feature count to image area (~1 per 1500 px)."""
    return max(3000, int(img.shape[0] * img.shape[1] / 1500))


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


@dataclass
class MapIndex:
    """Pre-extracted ORB features for a reference map."""
    points_px: np.ndarray   # (N, 2) float32
    descriptors: np.ndarray  # (N, 32) uint8 for ORB
    map_shape: tuple[int, int]  # (height, width)

    @classmethod
    def build(cls, map_img: np.ndarray, nfeatures: int | None = None) -> MapIndex:
        gray = _to_gray(map_img)
        if nfeatures is None:
            nfeatures = _default_nfeatures(gray)
        detector = cv2.ORB.create(nfeatures=nfeatures)
        kps, des = detector.detectAndCompute(gray, None)
        if des is None:
            pts = np.empty((0, 2), dtype=np.float32)
            des = np.empty((0, 32), dtype=np.uint8)
        else:
            pts = np.array([kp.pt for kp in kps], dtype=np.float32)
        return cls(points_px=pts, descriptors=des, map_shape=gray.shape[:2])

    def query(
        self, center_px: tuple[float, float] | None = None,
        radius_px: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (points, descriptors) inside a square window, or all."""
        if center_px is None or radius_px is None:
            return self.points_px, self.descriptors
        cx, cy = center_px
        mask = (
            (self.points_px[:, 0] >= cx - radius_px) &
            (self.points_px[:, 0] <= cx + radius_px) &
            (self.points_px[:, 1] >= cy - radius_px) &
            (self.points_px[:, 1] <= cy + radius_px)
        )
        return self.points_px[mask], self.descriptors[mask]


def _get_map_index(map_img: np.ndarray) -> MapIndex:
    """One-slot identity cache: reuse the index if the array is the same object."""
    key = id(map_img)
    cached = _map_cache.get(key)
    if cached is not None:
        return cached
    _map_cache.clear()
    idx = MapIndex.build(map_img)
    _map_cache[key] = idx
    return idx


def _match_core(
    frame: np.ndarray,
    map_source: MapIndex | np.ndarray,
    prior_px: tuple[float, float] | None = None,
    search_radius_px: float | None = None,
):
    """Detect, match, and fit a similarity transform (frame -> map).

    Returns (M, inliers, pts_frame, pts_map, n_total) or None.
    """
    if isinstance(map_source, np.ndarray):
        mi = _get_map_index(map_source)
    else:
        mi = map_source

    map_pts, map_des = mi.query(prior_px, search_radius_px)
    if len(map_des) < 2:
        return None

    frame_gray = _to_gray(frame)
    frame_det = cv2.ORB.create(nfeatures=FRAME_NFEATURES)
    kp_frame, des_frame = frame_det.detectAndCompute(frame_gray, None)

    if des_frame is None or len(des_frame) < 2:
        return None

    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des_frame, map_des, k=2)
    good = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 10:
        return None

    pts_frame = np.float32([kp_frame[m.queryIdx].pt for m in good])
    pts_map = np.float32([map_pts[m.trainIdx] for m in good])

    M, inliers = cv2.estimateAffinePartial2D(pts_frame, pts_map)
    if M is None or inliers is None:
        return None

    return M, inliers, pts_frame, pts_map, len(good)


def match_frame(
    frame: np.ndarray,
    map_img: np.ndarray | MapIndex,
    detector=None,  # kept for signature compat, ignored
    prior_px: tuple[float, float] | None = None,
    search_radius_px: float | None = None,
):
    """Find the frame's centre position on the map in pixel coordinates.

    Returns (px_x, px_y) as a numpy array, or None on failure.
    """
    result = _match_core(frame, map_img, prior_px, search_radius_px)
    if result is None:
        return None

    M = result[0]
    h, w = frame.shape[:2]
    return M @ np.array([w / 2, h / 2, 1])


def estimate_pose(
    frame: np.ndarray,
    map_img: np.ndarray | MapIndex,
    georef: MapGeoref,
    detector=None,  # kept for signature compat, ignored
    sigma_lut: list[dict] | None = None,
    prior_px: tuple[float, float] | None = None,
    search_radius_px: float | None = None,
    expected_scale: float | None = None,
):
    """Estimate position, heading, and calibrated uncertainty of a frame.

    Returns a Fix with lat/lon, heading, sigma_m, and ok flag.
    Reports Fix(ok=False) when match quality is insufficient -- a
    confidently wrong position is worse than no answer.
    """
    result = _match_core(frame, map_img, prior_px, search_radius_px)
    if result is None:
        return _no_fix()

    M, inliers, pts_frame, pts_map, n_total = result

    h, w = frame.shape[:2]
    center_px = M @ np.array([w / 2, h / 2, 1])

    if isinstance(map_img, MapIndex):
        map_h, map_w = map_img.map_shape
    else:
        map_h, map_w = map_img.shape[:2]
    if not (0 <= center_px[0] < map_w and 0 <= center_px[1] < map_h):
        return _no_fix()

    lat, lon = px_to_latlon(georef, float(center_px[0]), float(center_px[1]))

    scale = float(np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2))
    if scale < 0.1 or scale > 10:
        return _no_fix()
    if expected_scale is not None:
        if abs(scale - expected_scale) > 0.35 * expected_scale:
            return _no_fix()
    heading_rad = float(np.arctan2(M[1, 0], M[0, 0]))

    inlier_mask = inliers.ravel().astype(bool)
    n_inliers = int(inlier_mask.sum())
    if n_inliers < 4:
        return _no_fix()

    pts_h = np.hstack([pts_frame[inlier_mask], np.ones((n_inliers, 1))])
    predicted = (M @ pts_h.T).T
    residuals = predicted - pts_map[inlier_mask]
    rms_px = float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))

    avg_gsd = (georef.gsd_x + georef.gsd_y) / 2
    sigma_m = max(rms_px * avg_gsd / np.sqrt(n_inliers), avg_gsd)

    inlier_ratio = n_inliers / n_total

    if sigma_lut is not None:
        sigma_m, ok = _sigma_from_lut(inlier_ratio, sigma_lut)
        if not ok:
            return _no_fix()
    else:
        if inlier_ratio < 0.25:
            return _no_fix()
        sigma_m = float(np.interp(inlier_ratio, [0.25, 0.35, 0.50, 1.00],
                                                 [1.00, 0.50, 0.20, 0.20]))

    return Fix(
        lat=lat, lon=lon, heading_rad=heading_rad, sigma_m=sigma_m, ok=True,
        n_inliers=n_inliers, inlier_ratio=inlier_ratio,
    )
