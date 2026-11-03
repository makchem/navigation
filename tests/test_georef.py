"""Round-trip tests for the coordinate conversion layer."""

import pytest
from navmod.georef import (
    MapGeoref,
    px_to_enu,
    enu_to_px,
    px_to_latlon,
    latlon_to_px,
    latlon_to_enu,
    enu_to_latlon,
    distance_m,
)

# A ~4 km x ~3.5 km patch of Dresden — realistic for the test map.
GEOREF = MapGeoref(
    top_left_lat=51.065,
    top_left_lon=13.720,
    bottom_right_lat=51.033,
    bottom_right_lon=13.785,
    image_width_px=8192,
    image_height_px=5074,
)


class TestMapGeoref:
    def test_ground_dimensions_plausible(self):
        w = GEOREF.ground_width_m
        h = GEOREF.ground_height_m
        assert 3500 < w < 5500, f"ground width {w:.0f} m out of range"
        assert 3000 < h < 4500, f"ground height {h:.0f} m out of range"

    def test_gsd_positive(self):
        assert GEOREF.gsd_x > 0
        assert GEOREF.gsd_y > 0

    def test_gsd_submetre(self):
        assert GEOREF.gsd_x < 1.0
        assert GEOREF.gsd_y < 1.0

    def test_rejects_inverted_lat(self):
        with pytest.raises(ValueError, match="north"):
            MapGeoref(
                top_left_lat=51.0, top_left_lon=13.7,
                bottom_right_lat=51.1, bottom_right_lon=13.8,
                image_width_px=100, image_height_px=100,
            )

    def test_rejects_zero_image(self):
        with pytest.raises(ValueError, match="positive"):
            MapGeoref(
                top_left_lat=51.1, top_left_lon=13.7,
                bottom_right_lat=51.0, bottom_right_lon=13.8,
                image_width_px=0, image_height_px=100,
            )


class TestPixelEnuRoundTrip:
    @pytest.mark.parametrize("px_x,px_y", [
        (0, 0),
        (4096, 2537),
        (8191, 5073),
        (1000.5, 2000.5),
    ])
    def test_roundtrip(self, px_x, px_y):
        east, north = px_to_enu(GEOREF, px_x, px_y)
        rx, ry = enu_to_px(GEOREF, east, north)
        assert rx == pytest.approx(px_x, abs=1e-6)
        assert ry == pytest.approx(px_y, abs=1e-6)

    def test_origin_is_zero(self):
        e, n = px_to_enu(GEOREF, 0, 0)
        assert e == pytest.approx(0.0)
        assert n == pytest.approx(0.0)

    def test_east_positive_rightward(self):
        e1, _ = px_to_enu(GEOREF, 100, 0)
        e2, _ = px_to_enu(GEOREF, 200, 0)
        assert e2 > e1

    def test_north_negative_downward(self):
        _, n1 = px_to_enu(GEOREF, 0, 100)
        _, n2 = px_to_enu(GEOREF, 0, 200)
        assert n2 < n1


class TestPixelLatlonRoundTrip:
    @pytest.mark.parametrize("px_x,px_y", [
        (0, 0),
        (4096, 2537),
        (8192, 5074),
        (1234.5, 4321.0),
    ])
    def test_roundtrip(self, px_x, px_y):
        lat, lon = px_to_latlon(GEOREF, px_x, px_y)
        rx, ry = latlon_to_px(GEOREF, lat, lon)
        assert rx == pytest.approx(px_x, abs=1e-6)
        assert ry == pytest.approx(px_y, abs=1e-6)

    def test_topleft_corner(self):
        lat, lon = px_to_latlon(GEOREF, 0, 0)
        assert lat == pytest.approx(GEOREF.top_left_lat, abs=1e-9)
        assert lon == pytest.approx(GEOREF.top_left_lon, abs=1e-9)

    def test_bottomright_corner(self):
        lat, lon = px_to_latlon(GEOREF, GEOREF.image_width_px, GEOREF.image_height_px)
        assert lat == pytest.approx(GEOREF.bottom_right_lat, abs=1e-9)
        assert lon == pytest.approx(GEOREF.bottom_right_lon, abs=1e-9)


class TestLatlonEnuRoundTrip:
    @pytest.mark.parametrize("lat,lon", [
        (51.065, 13.720),
        (51.050, 13.750),
        (51.033, 13.785),
    ])
    def test_roundtrip(self, lat, lon):
        e, n = latlon_to_enu(GEOREF, lat, lon)
        rlat, rlon = enu_to_latlon(GEOREF, e, n)
        assert rlat == pytest.approx(lat, abs=1e-6)
        assert rlon == pytest.approx(lon, abs=1e-6)


class TestDistance:
    def test_zero(self):
        assert distance_m(51.05, 13.75, 51.05, 13.75) == pytest.approx(0.0)

    def test_known_distance(self):
        # ~1 degree of latitude is ~111 km
        d = distance_m(51.0, 13.0, 52.0, 13.0)
        assert 110_000 < d < 112_000

    def test_symmetric(self):
        d1 = distance_m(51.0, 13.0, 51.1, 13.1)
        d2 = distance_m(51.1, 13.1, 51.0, 13.0)
        assert d1 == pytest.approx(d2, abs=1e-6)
