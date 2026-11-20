"""Synthetic drone views cropped from the reference map.

This is the ground-truth generator: you know exactly which (x, y, heading,
altitude) a frame came from, so localization error is measurable rather than
eyeballed. Build this first — the other modules get tested against it.
"""
from __future__ import annotations

import cv2

from navmod.augment import AugmentConfig, augment_frame


def render_frame(
    map_img,
    x: float,
    y: float,
    heading_deg: float = 0.0,
    scale: float = 1.0,
    size: tuple[int, int] = (512, 512),
    augment: AugmentConfig | bool = False,
):
    """Crop a synthetic nadir frame centred on map pixel (x, y).

    `scale` stands in for altitude: >1 sees more ground at lower detail.

    ``augment`` controls realistic degradation so the frame is not pixel-
    identical to the reference map:
      - ``False`` / ``None``: raw crop (legacy, easy-mode matching).
      - ``True``: apply default AugmentConfig.
      - ``AugmentConfig(...)``: apply custom parameters.
    """
    M = cv2.getRotationMatrix2D(center=(x, y),
                                angle=heading_deg,
                                scale=1.0 / scale)
    M[0, 2] += size[0] / 2 - x
    M[1, 2] += size[1] / 2 - y
    frame = cv2.warpAffine(map_img, M, size)

    if augment is True:
        frame = augment_frame(frame)
    elif isinstance(augment, AugmentConfig):
        frame = augment_frame(frame, augment)

    return frame


# Old name kept for notebook compatibility during transition
fake_drone_frame = render_frame
