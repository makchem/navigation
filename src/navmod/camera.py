"""Synthetic drone views cropped from the reference map.

This is the ground-truth generator: you know exactly which (x, y, heading,
altitude) a frame came from, so localization error is measurable rather than
eyeballed. Build this first — the other modules get tested against it.
"""


def render_frame(
    map_img,
    x: float,
    y: float,
    heading_deg: float = 0.0,
    scale: float = 1.0,
    size: tuple[int, int] = (512, 512),
):
    """Crop a synthetic nadir frame centred on map pixel (x, y).

    `scale` stands in for altitude: >1 sees more ground at lower detail.
    Returns the frame; consider also returning the transform you used, so
    tests can compare against it directly.
    """
    raise NotImplementedError


# Old name kept for notebook compatibility during transition
fake_drone_frame = render_frame
