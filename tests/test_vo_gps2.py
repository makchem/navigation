"""Tests for SITL-only GPS2 adapter: absolute anchor, loss and acceptance gates."""
import math
from types import SimpleNamespace

from navmod.georef import MapGeoref, enu_to_latlon, latlon_to_enu
from navmod.vo_gps2 import VOGps2Adapter

GEO = MapGeoref(51.0800079, 13.6911106, 51.0575373, 13.7485796, 8192, 5097)
LAT, LON = 51.069, 13.720


def gps(on=True):
    return SimpleNamespace(available=on, lat=LAT, lon=LON)


def matcher(ok=False):
    return SimpleNamespace(ok=ok, lat=LAT, lon=LON)


def row(segment, status, e=0.0, n=0.0, sigma=1.0, ve=8, vn=4):
    return dict(segment_id=segment, status=status,
                position_east_m=e, position_north_m=n, sigma_m=sigma,
                ve_mps=ve, vn_mps=vn)


def test_gps_bootstrap_then_vo_only():
    adapter = VOGps2Adapter(GEO)
    assert adapter.update(vo_row=row(0, 'INIT'), gps=gps(), matcher_fix=matcher()) is None
    # While GPS1 is enabled the anchor tracks its position.
    assert adapter.update(vo_row=row(0, 'OK', 3, 4), gps=gps(), matcher_fix=matcher())
    out = adapter.update(vo_row=row(0, 'OK', 8, 1), gps=gps(False), matcher_fix=matcher())
    assert out is not None and out.anchor_source == 'GPS1'
    east0, north0 = latlon_to_enu(GEO, LAT, LON)
    east, north = latlon_to_enu(GEO, out.lat, out.lon)
    assert abs(east - east0 - 5) < 1e-5
    assert abs(north - north0 + 3) < 1e-5
    assert out.reported_accuracy_m >= 25


def test_can_start_from_matcher_without_gps():
    a = VOGps2Adapter(GEO)
    assert a.update(vo_row=row(4, 'INIT'), gps=gps(False), matcher_fix=matcher()) is None
    assert a.update(vo_row=row(4, 'OK', 11, 5), gps=gps(False), matcher_fix=matcher()) is None
    fix = a.update(vo_row=row(4, 'OK', 13, 5), gps=gps(False), matcher_fix=matcher(True))
    assert fix is not None and fix.anchor_source == 'MATCHER'
    assert math.isclose(fix.lat, LAT, abs_tol=1e-9)
    assert math.isclose(fix.lon, LON, abs_tol=1e-9)


def test_loss_breaks_anchor_and_recovery_needs_new_anchor():
    a = VOGps2Adapter(GEO)
    a.update(vo_row=row(0, 'INIT'), gps=gps(), matcher_fix=matcher())
    assert a.update(vo_row=row(0, 'OK', 6, 4), gps=gps(False), matcher_fix=matcher())
    assert a.update(vo_row=row(0, 'LOST'), gps=gps(False), matcher_fix=matcher()) is None
    assert a.update(vo_row=row(1, 'INIT'), gps=gps(False), matcher_fix=matcher()) is None
    assert a.update(vo_row=row(1, 'OK', 4, 3), gps=gps(False), matcher_fix=matcher()) is None
    assert a.update(vo_row=row(1, 'OK', 8, 5), gps=gps(False), matcher_fix=matcher(True)) is not None


def test_does_not_extrapolate_when_no_frame_or_uncertain():
    a = VOGps2Adapter(GEO)
    a.update(vo_row=row(0, 'INIT'), gps=gps(), matcher_fix=matcher())
    assert a.update(vo_row=None, gps=gps(False), matcher_fix=matcher()) is None
    assert a.update(vo_row=row(0, 'OK', 4, 2, sigma=40), gps=gps(False), matcher_fix=matcher()) is None
    assert a.update(vo_row=row(0, 'OK', 4, 2, sigma=1, ve=120), gps=gps(False), matcher_fix=matcher()) is None
    assert a.update(vo_row=row(0, 'OK', 4, 2, sigma=1), gps=gps(False), matcher_fix=matcher()) is not None
