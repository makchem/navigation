
import cv2
import numpy as np
import math

from navmod.odometry import camera_gsd, pixel_shift_to_enu
from navmod.odometry import VisualOdometry
from navmod.types import CameraFrame


def test_known_pixel_shift():
    rng = np.random.default_rng(42)

    img1 = rng.integers(
        0, 255, (512, 512), dtype=np.uint8
    )
    img1 = cv2.GaussianBlur(img1, (5, 5), 0)

    # Зсуваємо зображення вправо на 6 px
    # і вниз на 4 px.
    transform = np.float32([
        [1, 0, 6],
        [0, 1, 4],
    ])

    img2 = cv2.warpAffine(
        img1, transform, (512, 512)
    )

    vo = VisualOdometry()

    first = vo.update(CameraFrame(
        image=img1,
        timestamp_s=1.0,
        height_agl_m=100.0,
        frame_id=0,
    ))

    second = vo.update(CameraFrame(
        image=img2,
        timestamp_s=1.2,
        height_agl_m=100.0,
        frame_id=1,
    ))

    assert not first.ok
    assert second.ok

    # Рух камери протилежний руху фону
    assert abs(second.dx_px + 6) < 1
    assert abs(second.dy_px + 4) < 1

def test_pixel_to_metres_north():
    # Камера на висоті 100 м, повернута на північ
    gsd = camera_gsd(100, 60, 512, 512)

    east, north = pixel_shift_to_enu(
        dx_px=-6,
        dy_px=-4,
        height_agl_m=100,
        yaw_rad=0,
        fov_diag_deg=60,
        width_px=512,
        height_px=512,
    )

    assert math.isclose(east, -6 * gsd, abs_tol=1e-6)
    assert math.isclose(north, 4 * gsd, abs_tol=1e-6)


def test_pixel_to_metres_east():
    # Камера повернута на схід (yaw = 90°)
    gsd = camera_gsd(100, 60, 512, 512)

    east, north = pixel_shift_to_enu(
        dx_px=-6,
        dy_px=-4,
        height_agl_m=100,
        yaw_rad=math.pi / 2,
        fov_diag_deg=60,
        width_px=512,
        height_px=512,
    )

    assert math.isclose(east, 4 * gsd, abs_tol=1e-6)
    assert math.isclose(north, 6 * gsd, abs_tol=1e-6)

def test_metric_velocity():
    rng = np.random.default_rng(42)

    img1 = rng.integers(
        0, 255, (512, 512), dtype=np.uint8
    )
    img1 = cv2.GaussianBlur(img1, (5, 5), 0)

    # Фон зміщується на 6 px вправо і 4 вниз
    M = np.float32([
        [1, 0, 6],
        [0, 1, 4],
    ])

    img2 = cv2.warpAffine(img1, M, (512, 512))

    vo = VisualOdometry(fov_diag_deg=60)

    vo.update(CameraFrame(
        image=img1,
        timestamp_s=1.0,
        height_agl_m=100.0,
        frame_id=0,
    ))

    result = vo.update(
        CameraFrame(
            image=img2,
            timestamp_s=1.2,
            height_agl_m=100.0,
            frame_id=1,
        ),
        yaw_rad=0.0,
    )

    gsd = camera_gsd(100, 60, 512, 512)

    assert result.ok

    assert abs(result.delta_east_m - (-6 * gsd)) < gsd
    assert abs(result.delta_north_m - (4 * gsd)) < gsd

    assert abs(result.ve_mps - (-6 * gsd / 0.2)) < gsd / 0.2
    assert abs(result.vn_mps - (4 * gsd / 0.2)) < gsd / 0.2



from navmod.camera import render_frame


def test_visual_yaw_rotation():
    rng = np.random.default_rng(42)

    map_img = rng.integers(
        0, 255, (900, 900), dtype=np.uint8
    )
    map_img = cv2.GaussianBlur(
        map_img, (5, 5), 0
    )

    # Камера повертається на 3 градуси,
    # але її координати не змінюються.
    img0 = render_frame(
        map_img, 450, 450,
        heading_deg=0,
        size=(512, 512),
    )
    img1 = render_frame(
        map_img, 450, 450,
        heading_deg=3,
        size=(512, 512),
    )

    vo = VisualOdometry()

    vo.update(CameraFrame(
        image=img0,
        timestamp_s=1.0,
        height_agl_m=100,
        frame_id=0,
    ))

    result = vo.update(CameraFrame(
        image=img1,
        timestamp_s=1.2,
        height_agl_m=100,
        frame_id=1,
    ))

    assert result.ok
    assert abs(
        math.degrees(result.delta_yaw_rad) - 3
    ) < 0.5
