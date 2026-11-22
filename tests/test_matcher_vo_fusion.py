"""Matcher-first absolute correction + VO fallback tests."""
from types import SimpleNamespace
import math
import pytest
from navmod.georef import MapGeoref, enu_to_latlon, latlon_to_enu
from navmod.matcher_vo_fusion import MatcherFirstFusion

GEO = MapGeoref(51.0800079,13.6911106,51.0575373,13.7485796,8192,5097)
HOME = (51.069, 13.720)


def gps(on=True):
    return SimpleNamespace(available=on, lat=HOME[0], lon=HOME[1])


def make_row(t, fid, seg=0, e=0.0, n=0.0, status='OK'):
    return {'frame_id':fid,'t_sim':t,'segment_id':seg,'status':status,
            'position_east_m':e, 'position_north_m':n}


def matched(e_offset=0.0,n_offset=0.0):
    e,n=latlon_to_enu(GEO,*HOME)
    lat,lon=enu_to_latlon(GEO,e+e_offset,n+n_offset)
    return SimpleNamespace(lat=lat,lon=lon,ok=True,n_inliers=30,inlier_ratio=.7)


def test_vo_fallback_without_matcher():
    f=MatcherFirstFusion(GEO)
    f.seed_origin(gps())
    f.observe_frame(make_row(10,1,status='INIT'),gps())
    f.observe_frame(make_row(10.2,2,e=4,n=7),gps(False))
    out=f.fused_ned(make_row(10.2,2,e=4,n=7))
    assert out[:2] == pytest.approx((7,4),abs=1e-5)
    assert out[2]=='vo'


def test_matcher_priority_and_continuous_vo_fallback():
    f=MatcherFirstFusion(GEO,correction_mps=5)
    f.seed_origin(gps())
    s=f.observe_frame(make_row(1.0,0, status='INIT'),gps())
    f.observe_frame(make_row(1.2,1,e=4),gps(False))
    assert f.fused_ned(make_row(1.2,1,e=4))[1] == pytest.approx(4)
    # Late matcher fix for frame 1 measures +6 m rather than +4 m.
    late=f.accept_match(matched(6), f.observe_frame(make_row(1.2,1,e=4),gps(False)),1.4)
    assert late.accepted
    f.observe_frame(make_row(1.4,2,e=8),gps(False))
    a=f.fused_ned(make_row(1.4,2,e=8))
    assert a[2]=='matcher'
    # Bounded correction 5m/s * .2 s = +1m; no full instant jump.
    assert a[1] == pytest.approx(9.0, abs=.1)
    f.observe_frame(make_row(1.6,3,e=12),gps(False))
    b=f.fused_ned(make_row(1.6,3,e=12))
    assert b[1] == pytest.approx(14.0, abs=.1)
    f.observe_frame(make_row(3.5,4,e=20),gps(False))
    c=f.fused_ned(make_row(3.5,4,e=20))
    assert c[2]=='vo'


def test_late_match_uses_old_frame_vo_position():
    f=MatcherFirstFusion(GEO,correction_mps=100)
    f.seed_origin(gps())
    f.observe_frame(make_row(9.8,9,status="INIT",e=0),gps())
    old=f.observe_frame(make_row(10,10, e=20),gps(False))
    f.observe_frame(make_row(10.2,11,e=30),gps(False))
    f.fused_ned(make_row(10.2,11,e=30))
    # Matcher observes absolute 25m at old frame (VO reads 20m).
    d=f.accept_match(matched(25),old,10.3)
    assert d.accepted
    f.observe_frame(make_row(10.4,12,e=40),gps(False))
    # VO carries forward: corrected current position is ~45m.
    out=f.fused_ned(make_row(10.4,12,e=40))
    assert out[1]==pytest.approx(45.0,abs=1e-5)


def test_loss_requires_new_anchor_and_old_match_rejected():
    f=MatcherFirstFusion(GEO)
    f.seed_origin(gps())
    old=f.observe_frame(make_row(1,0,status='INIT'),gps())
    f.observe_frame(make_row(1.2,1,status='LOST'),gps(False))
    f.observe_frame(make_row(1.4,2,seg=1,status='INIT'),gps(False))
    f.observe_frame(make_row(1.6,3,seg=1,e=3),gps(False))
    assert f.fused_ned(make_row(1.6,3,seg=1,e=3)) is None
    assert not f.accept_match(matched(0),old,1.6).accepted
    new=f.observe_frame(make_row(1.8,4,seg=1,e=6),gps(False))
    assert f.accept_match(matched(25),new,1.9).accepted
    out=f.fused_ned(make_row(1.8,4,seg=1,e=6))
    assert out[1]==pytest.approx(25.0,abs=1e-5)


def test_outlier_match_does_not_teleport():
    f=MatcherFirstFusion(GEO,max_innovation_m=40)
    f.seed_origin(gps())
    snap=f.observe_frame(make_row(1,0,e=0),gps())
    assert f.fused_ned(make_row(1,0,e=0))[1]==pytest.approx(0)
    d=f.accept_match(matched(300),snap,1.1)
    assert not d.accepted and d.reason=='MATCH_OUTLIER_WAIT_RECONFIRM'
    f.observe_frame(make_row(1.2,1,e=8),gps(False))
    assert f.fused_ned(make_row(1.2,1,e=8))[1]==pytest.approx(8)


def test_stale_match_is_rejected():
    f=MatcherFirstFusion(GEO,max_match_age_s=2)
    snap=f.observe_frame(make_row(1,0),gps())
    assert f.accept_match(matched(0),snap,5).reason=='STALE_MATCH'


def test_large_correction_requires_two_consistent_matches():
    f=MatcherFirstFusion(GEO,max_innovation_m=40,correction_mps=10)
    f.seed_origin(gps())
    first=f.observe_frame(make_row(1,0,status='INIT'),gps())
    assert f.accept_match(matched(90),first,1.1).reason=='MATCH_OUTLIER_WAIT_RECONFIRM'
    second=f.observe_frame(make_row(1.2,1,e=6),gps(False))
    # Absolute 96m on the next frame means the same +90m offset.
    assert f.accept_match(matched(96),second,1.4).accepted
    out=f.fused_ned(make_row(1.2,1,e=6))
    assert out[2]=='matcher'
