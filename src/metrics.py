"""Ground-sampling-distance and error metrics.

Everything here converts between pixels and metres. Keep it dependency-light
so notebooks can import it without pulling in the matching stack.
"""


def gsd(ground_width_m: float, image_width_px: int) -> float:
    """Ground sampling distance in metres per pixel."""
    raise NotImplementedError


def error_m(estimated_xy, true_xy, gsd_m_per_px: float) -> float:
    """Euclidean localization error, in metres, between two pixel positions."""
    raise NotImplementedError
