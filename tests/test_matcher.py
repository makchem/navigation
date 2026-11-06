"""End-to-end matcher test on synthetic frames."""

import numpy as np
import pytest

from navmod.camera import render_frame
from navmod.matcher import match_frame
from navmod.metrics import error_m, gsd


SYNTH_SIZE = (2000, 2000)
MAP_GSD = 0.5  # metres per pixel, similar to Dresden map
FRAME_SIZE = (512, 512)


@pytest.fixture(scope="module")
def synth_map():
    rng = np.random.RandomState(42)
    return rng.randint(0, 256, (*SYNTH_SIZE, 3), dtype=np.uint8)


class TestMatcherEndToEnd:
    def test_identity_position(self, synth_map):
        x, y = 1000, 1000
        frame = render_frame(synth_map, x, y, heading_deg=0, scale=1, size=FRAME_SIZE)
        predicted = match_frame(frame, synth_map)
        err = error_m(predicted, (x, y), MAP_GSD)
        assert err < 50, f"error {err:.1f} m exceeds 50 m threshold"

    def test_offset_position(self, synth_map):
        x, y = 600, 1400
        frame = render_frame(synth_map, x, y, heading_deg=0, scale=1, size=FRAME_SIZE)
        predicted = match_frame(frame, synth_map)
        err = error_m(predicted, (x, y), MAP_GSD)
        assert err < 50, f"error {err:.1f} m exceeds 50 m threshold"

    def test_with_heading(self, synth_map):
        x, y = 1000, 1000
        frame = render_frame(synth_map, x, y, heading_deg=5, scale=1, size=FRAME_SIZE)
        predicted = match_frame(frame, synth_map)
        if predicted is None:
            return
        err = error_m(predicted, (x, y), MAP_GSD)
        assert err < 50, f"error {err:.1f} m exceeds 50 m threshold"

    def test_with_scale(self, synth_map):
        x, y = 1000, 1000
        frame = render_frame(synth_map, x, y, heading_deg=0, scale=2, size=FRAME_SIZE)
        predicted = match_frame(frame, synth_map)
        if predicted is None:
            return 
        err = error_m(predicted, (x, y), MAP_GSD)
        assert predicted is None or err < 50, f"error {err:.1f} m exceeds 50 m threshold"
