"""Ground-sampling-distance and error metrics.

Everything here converts between pixels and metres. Keep it dependency-light
so notebooks can import it without pulling in the matching stack.
"""


# --- ground sampling distance -------------------------------------------------
#
# Two ways to get GSD, used at opposite ends of the pipeline:
#
#   gsd()             reference map — you know the ground width it spans
#   gsd_from_camera() drone frame — you know altitude, focal length, sensor
#
# Both return metres per pixel. Derive GSD from *width*, not height: a
# truncated or cropped-short image keeps its full width, so a width-based
# figure survives what a height-based one would silently corrupt.


import numpy as np

def gsd(ground_width_m: float, image_width_px: int) -> float:
    """Ground sampling distance in metres per pixel.

    >>> round(gsd(5000, 8192), 4)
    0.6104
    """
    if ground_width_m <= 0 or image_width_px <= 0:
        raise ValueError(f"need positive inputs, got {ground_width_m=}, {image_width_px=}")
    return ground_width_m / image_width_px


def gsd_from_image(img, ground_width_m: float) -> float:
    """GSD of a loaded image, taking the pixel width from the array itself.

    `img` is any (H, W) or (H, W, C) array — the shape convention OpenCV and
    numpy share.
    """
    if getattr(img, "ndim", 0) < 2:
        raise TypeError(f"expected a 2D+ image array, got {type(img).__name__}")
    return gsd(ground_width_m, img.shape[1])


def gsd_from_camera(
    altitude_m: float,
    focal_length_mm: float,
    sensor_width_mm: float,
    image_width_px: int,
) -> float:
    """GSD of a nadir frame from camera geometry, in metres per pixel.

    The photogrammetry form: sensor and focal length must share units (mm);
    altitude carries the output unit (m).

    >>> round(gsd_from_camera(100, 8.8, 13.2, 5472), 4)  # ~2.7 cm/px
    0.0274
    """
    if min(altitude_m, focal_length_mm, sensor_width_mm) <= 0 or image_width_px <= 0:
        raise ValueError("all arguments must be positive")
    return (sensor_width_mm * altitude_m) / (focal_length_mm * image_width_px)

def error_m(estimated_xy, true_xy, gsd_m_per_px: float) -> float:
    """Euclidean localization error, in metres, between two pixel positions."""
    
    dist_px = np.linalg.norm(np.array(estimated_xy) - np.array(true_xy))
    return float(dist_px * gsd_m_per_px)
