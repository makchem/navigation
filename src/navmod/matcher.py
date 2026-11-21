"""Frame-to-map matching and pose recovery.

Reminder from the roadmap: prefer cv2.estimateAffinePartial2D over
findHomography for nadir imagery. A downward-looking camera over flat terrain
is a 4-DOF similarity (rotation, scale, translation); fitting 8 DOF spends the
extra freedom on noise and can produce confident nonsense on thin inlier sets.
"""

import cv2
import numpy as np

from navmod.georef import MapGeoref, px_to_latlon
from navmod.types import Fix

_NO_FIX = Fix(lat=0.0, lon=0.0, heading_rad=0.0, sigma_m=float("inf"), ok=False)


def _sigma_from_lut(inlier_ratio: float, lut: list[dict]) -> tuple[float, bool]:
    """Return (sigma_m, ok) from the lookup table."""
    for entry in lut:
        if inlier_ratio < entry["max_ratio"]:
            if not entry.get("ok", True):
                return 0.0, False
            return entry["sigma_m"], True
    last = lut[-1]
    return last["sigma_m"], True


def _default_nfeatures(img):
    """Scale ORB feature count to image area (~1 per 1500 px)."""
    return max(3000, int(img.shape[0] * img.shape[1] / 1500))


def _match_core(frame, map_img, detector=None):
    """Detect, match, and fit a similarity transform (frame → map).

    Returns (M, inliers, pts_frame, pts_map) or None.
    """
    if detector is None:
        detector = cv2.ORB.create(nfeatures=_default_nfeatures(map_img))

    kp_map, des_map = detector.detectAndCompute(map_img, None)
    kp_frame, des_frame = detector.detectAndCompute(frame, None)

    if des_frame is None or des_map is None:
        return None
    if len(des_frame) < 2 or len(des_map) < 2:
        return None

    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des_frame, des_map, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good) < 10:
        return None

    pts_frame = np.float32([kp_frame[m.queryIdx].pt for m in good])
    pts_map = np.float32([kp_map[m.trainIdx].pt for m in good])

    M, inliers = cv2.estimateAffinePartial2D(pts_frame, pts_map)
    if M is None or inliers is None:
        return None

    return M, inliers, pts_frame, pts_map, len(good)


def match_frame(frame, map_img, detector=None):
    """Find the frame's centre position on the map in pixel coordinates.

    Returns (px_x, px_y) as a numpy array, or None on failure.
    """
    result = _match_core(frame, map_img, detector)
    if result is None:
        return None

    M = result[0]  # (M, inliers, pts_frame, pts_map, n_total)
    h, w = frame.shape[:2]
    return M @ np.array([w / 2, h / 2, 1])


def estimate_pose(frame, map_img, georef: MapGeoref, detector=None,
                   sigma_lut: list[dict] | None = None):
    """Estimate position, heading, and calibrated uncertainty of a frame.

    Returns a Fix with lat/lon, heading, sigma_m, and ok flag.
    Reports Fix(ok=False) when match quality is insufficient — a
    confidently wrong position is worse than no answer.
    """
    result = _match_core(frame, map_img, detector)
    if result is None:
        return _NO_FIX

    M, inliers, pts_frame, pts_map, n_total = result

    # --- position: frame centre → map pixel → lat/lon ---
    h, w = frame.shape[:2]
    center_px = M @ np.array([w / 2, h / 2, 1])

    map_h, map_w = map_img.shape[:2]
    if not (0 <= center_px[0] < map_w and 0 <= center_px[1] < map_h):
        return _NO_FIX

    lat, lon = px_to_latlon(georef, float(center_px[0]), float(center_px[1]))

    # --- heading: rotation component of the recovered similarity ---
    # The recovered affine (frame→map) inverts the render transform,
    # so M[1,0] = s·sin(θ), M[0,0] = s·cos(θ) where θ = heading.
    scale = float(np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2))
    if scale < 0.1 or scale > 10:
        return _NO_FIX
    heading_rad = float(np.arctan2(M[1, 0], M[0, 0]))

    # --- calibrated uncertainty from reprojection residuals ---
    inlier_mask = inliers.ravel().astype(bool)
    n_inliers = int(inlier_mask.sum())
    if n_inliers < 4:
        return _NO_FIX

    pts_h = np.hstack([pts_frame[inlier_mask], np.ones((n_inliers, 1))])
    predicted = (M @ pts_h.T).T
    residuals = predicted - pts_map[inlier_mask]
    rms_px = float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))

    avg_gsd = (georef.gsd_x + georef.gsd_y) / 2
    # standard error of centre estimate ≈ rms / √n, floored at one pixel
    sigma_m = max(rms_px * avg_gsd / np.sqrt(n_inliers), avg_gsd)

    inlier_ratio = n_inliers / n_total

    if sigma_lut is not None:
        sigma_m, ok = _sigma_from_lut(inlier_ratio, sigma_lut)
        if not ok:
            return _NO_FIX
    else:
        if inlier_ratio < 0.25:
            return _NO_FIX
        sigma_m = float(np.interp(inlier_ratio, [0.25, 0.35, 0.50, 1.00],
                                                 [1.00, 0.50, 0.20, 0.20]))

    return Fix(
        lat=lat, lon=lon, heading_rad=heading_rad, sigma_m=sigma_m, ok=True,
        n_inliers=n_inliers, inlier_ratio=inlier_ratio,
    )
