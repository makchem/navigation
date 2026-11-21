from dataclasses import dataclass, replace

import cv2
import numpy as np
import math

from navmod.types import CameraFrame



@dataclass
class FlowResult:
    ok: bool = False

    # Зміщення у пікселях
    dx_px: float = float("nan")
    dy_px: float = float("nan")
    dt_s: float = float("nan")

    # Якість Optical Flow
    n_tracked: int = 0
    n_inliers: int = 0
    inlier_ratio: float = 0.0
    rms_px: float = float("inf")

    # Переміщення у метрах
    delta_east_m: float = float("nan")
    delta_north_m: float = float("nan")

    # Швидкість у м/с
    ve_mps: float = float("nan")
    vn_mps: float = float("nan")

    # Relative position from the current VO origin (ENU)
    position_east_m: float = float("nan")
    position_north_m: float = float("nan")

    # Provisional, NOT calibrated uncertainties
    sigma_delta_m: float = float("inf")
    sigma_m: float = float("inf")



def camera_gsd(
    height_agl_m: float,
    fov_diag_deg: float,
    width_px: int,
    height_px: int,
) -> float:
    """Ground sampling distance in metres per image pixel."""

    if height_agl_m <= 0:
        raise ValueError("Camera height must be positive")

    if width_px <= 0 or height_px <= 0:
        raise ValueError("Invalid image dimensions")

    if not 0 < fov_diag_deg < 180:
        raise ValueError("Invalid FOV")

    diagonal_px = math.hypot(width_px, height_px)

    return (
        2 * height_agl_m
        * math.tan(math.radians(fov_diag_deg) / 2)
        / diagonal_px
    )


def pixel_shift_to_enu(
    dx_px: float,
    dy_px: float,
    height_agl_m: float,
    yaw_rad: float,
    fov_diag_deg: float,
    width_px: int,
    height_px: int,
) -> tuple[float, float]:
    """Convert ground-centre displacement to East/North metres.

    Assumes a level, downward-looking camera and flat terrain.
    Uses the image orientation of the previous frame.
    """

    gsd = camera_gsd(
        height_agl_m,
        fov_diag_deg,
        width_px,
        height_px,
    )

    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)

    east_m = gsd * (c * dx_px - s * dy_px)
    north_m = -gsd * (s * dx_px + c * dy_px)

    return east_m, north_m



class VisualOdometry:
    def __init__(self, fov_diag_deg: float = 60.0):
        self.fov_diag_deg = fov_diag_deg
        self.reset()

    def reset(
        self,
        position_east_m: float = 0.0,
        position_north_m: float = 0.0,
        sigma_m: float = 0.0,
    ) -> None:
        """Start a fresh track anchored at a known local ENU position."""
        values = (position_east_m, position_north_m, sigma_m)
        if not all(math.isfinite(v) for v in values) or sigma_m < 0:
            raise ValueError("Reset position and sigma must be finite; sigma >= 0")
        self.prev_gray = None
        self.prev_t = None
        self.prev_height_m = None
        self.position_east_m = float(position_east_m)
        self.position_north_m = float(position_north_m)
        self.sigma_m = float(sigma_m)
        self._tracking_valid = True

    def _estimate_flow(self, frame: CameraFrame, yaw_rad: float = 0.0) -> FlowResult:
        image = frame.image

        # Переводимо кадр у grayscale
        if image.ndim == 2:
            gray = image
        elif image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        prev, prev_t = self.prev_gray, self.prev_t
        prev_height = self.prev_height_m

        # Запам'ятовуємо поточний кадр
        self.prev_gray = gray.copy()
        self.prev_t = frame.timestamp_s
        self.prev_height_m = frame.height_agl_m

        if prev is None or prev.shape != gray.shape:
            return FlowResult()

        dt = frame.timestamp_s - prev_t

        if not 0 < dt <= 0.5:
            return FlowResult(dt_s=dt)

        # 1. Знаходимо характерні точки
        points = cv2.goodFeaturesToTrack(
            prev,
            maxCorners=600,
            qualityLevel=0.01,
            minDistance=8,
            blockSize=7,
        )

        if points is None or len(points) < 20:
            return FlowResult(dt_s=dt)

        # 2. Відстежуємо точки на новому кадрі
        tracked, status, _ = cv2.calcOpticalFlowPyrLK(
            prev, gray, points, None,
            winSize=(21, 21),
            maxLevel=3,
        )

        if tracked is None or status is None:
            return FlowResult(dt_s=dt)

        keep = status.ravel() == 1

        p0 = points.reshape(-1, 2)[keep]
        p1 = tracked.reshape(-1, 2)[keep]

        if len(p0) < 20:
            return FlowResult(
                dt_s=dt, n_tracked=len(p0)
            )

        # 3. Відкидаємо неправильні відповідності
        M, mask = cv2.estimateAffinePartial2D(
            p0, p1,
            method=cv2.RANSAC,
            ransacReprojThreshold=2.5,
        )

        if M is None or mask is None:
            return FlowResult(
                dt_s=dt, n_tracked=len(p0)
            )

        good = mask.ravel().astype(bool)
        n = int(good.sum())
        ratio = n / len(p0)

        if n < 15 or ratio < 0.55:
            return FlowResult(
                dt_s=dt,
                n_tracked=len(p0),
                n_inliers=n,
                inlier_ratio=ratio,
            )

        # 4. Знаходимо переміщення центру камери
        A = M[:, :2]
        t = M[:, 2]

        h, w = gray.shape
        center = np.array([w / 2, h / 2])

        if abs(np.linalg.det(A)) < 1e-6:
            return FlowResult(dt_s=dt)

        previous_center = np.linalg.solve(
            A, center - t
        )

        # 5. Обчислюємо помилку репроєкції
        predicted = (A @ p0[good].T).T + t

        rms = float(np.sqrt(np.mean(np.sum(
            (predicted - p1[good]) ** 2,
            axis=1
        ))))


        # 6. Зсув у пікселях
        dx_px = float(previous_center[0] - center[0])
        dy_px = float(previous_center[1] - center[1])

        # 7. Середня висота між кадрами
        height_m = (prev_height + frame.height_agl_m) / 2

        if not np.isfinite(height_m) or height_m <= 0:
            return FlowResult(dt_s=dt)

        if not np.isfinite(yaw_rad):
            return FlowResult(dt_s=dt)

        # 8. Переміщення в метрах
        delta_east, delta_north = pixel_shift_to_enu(
            dx_px=dx_px,
            dy_px=dy_px,
            height_agl_m=height_m,
            yaw_rad=yaw_rad,
            fov_diag_deg=self.fov_diag_deg,
            width_px=w,
            height_px=h,
        )

        # 9. Швидкість у м/с
        ve = delta_east / dt
        vn = delta_north / dt

        return FlowResult(
            ok=rms <= 2.5,
            dx_px=dx_px,
            dy_px=dy_px,
            dt_s=dt,
            n_tracked=len(p0),
            n_inliers=n,
            inlier_ratio=ratio,
            rms_px=rms,
            delta_east_m=delta_east,
            delta_north_m=delta_north,
            ve_mps=ve,
            vn_mps=vn,
        )


    def update(self, frame: CameraFrame, yaw_rad: float = 0.0) -> FlowResult:
        """Estimate frame-to-frame motion and integrate only uninterrupted valid steps."""
        had_previous = self.prev_gray is not None
        prev_height = self.prev_height_m
        flow = self._estimate_flow(frame, yaw_rad)

        # First image is the local origin, not a measured displacement.
        if not had_previous:
            return replace(
                flow,
                position_east_m=self.position_east_m,
                position_north_m=self.position_north_m,
                sigma_m=self.sigma_m,
            )

        # A missing/bad step breaks the chain. Do not silently bridge the gap.
        if not flow.ok:
            self._tracking_valid = False
            return replace(flow, sigma_m=float("inf"))
        if not self._tracking_valid:
            # We may have a good *relative* shift but no valid accumulated pose.
            return replace(flow, ok=False, sigma_m=float("inf"))

        height_m = (prev_height + frame.height_agl_m) / 2.0
        height_px, width_px = frame.image.shape[:2]
        gsd = camera_gsd(height_m, self.fov_diag_deg, width_px, height_px)
        step_m = math.hypot(flow.delta_east_m, flow.delta_north_m)

        # EXPERIMENTAL error-growth model; constants require SITL calibration.
        sigma_delta_m = math.sqrt(
            0.20**2
            + (2.0 * flow.rms_px * gsd)**2
            + (0.05 * step_m)**2
        )
        self.position_east_m += flow.delta_east_m
        self.position_north_m += flow.delta_north_m
        self.sigma_m = math.hypot(self.sigma_m, sigma_delta_m)

        return replace(
            flow,
            position_east_m=self.position_east_m,
            position_north_m=self.position_north_m,
            sigma_delta_m=sigma_delta_m,
            sigma_m=self.sigma_m,
        )
