import math
from types import SimpleNamespace
from navmod.vo_mavlink import ExternalNavPublisher


def row(t=1.0, segment=0, status='OK', **kwargs):
    r = dict(t_sim=t, segment_id=segment, status=status, vo_ok=True,
             position_east_m=20, position_north_m=10,
             ve_mps=8, vn_mps=4, sigma_m=1,
             inlier_ratio=0.8)
    r.update(kwargs)
    return r


def attitude(yaw=0, roll=0, pitch=0):
    return SimpleNamespace(yaw=yaw, roll=roll, pitch=pitch)


def test_frames_and_ned_position():
    pub = ExternalNavPublisher(min_good_frames=1)
    p = pub.payload(row(), attitude())
    assert (p['frame_id'], p['child_frame_id']) == (20, 12)
    assert (p['x'], p['y'], p['z']) == (10, 20, 0)
    assert abs(p['vx'] - 4) < 1e-6
    assert abs(p['vy'] - 8) < 1e-6
    assert len(p['pose_covariance']) == 21


def test_rotated_body_velocity():
    pub = ExternalNavPublisher(min_good_frames=1)
    p = pub.payload(row(), attitude(yaw=math.pi/2))
    # Facing east: forward is east; right is south.
    assert abs(p['vx'] - 8) < 1e-6
    assert abs(p['vy'] + 4) < 1e-6


def test_roll_pitch_and_yaw_rotate_body_velocity_back_to_ned():
    pub = ExternalNavPublisher(min_good_frames=1)
    p = pub.payload(row(), attitude(yaw=0.8, roll=0.22, pitch=-0.18))
    w, x, y, z = p['q']
    vx, vy, vz = p['vx'], p['vy'], p['vz']
    ned_n = (1-2*(y*y+z*z))*vx + 2*(x*y-w*z)*vy + 2*(x*z+w*y)*vz
    ned_e = 2*(x*y+w*z)*vx + (1-2*(x*x+z*z))*vy + 2*(y*z-w*x)*vz
    ned_d = 2*(x*z-w*y)*vx + 2*(y*z+w*x)*vy + (1-2*(x*x+y*y))*vz
    assert abs(ned_n-4) < 1e-6
    assert abs(ned_e-8) < 1e-6
    assert abs(ned_d) < 1e-6


def test_start_gate_loss_and_segment_reset():
    pub = ExternalNavPublisher(min_good_frames=2)
    assert pub.payload(row(1.0), attitude()) is None
    p1 = pub.payload(row(1.2), attitude())
    assert p1 is not None
    assert pub.payload(row(1.4, status='LOST'), attitude()) is None
    assert pub.payload(row(1.6, segment=1), attitude()) is None
    p2 = pub.payload(row(1.8, segment=1), attitude())
    assert p2['reset_counter'] == (p1['reset_counter'] + 1) % 256


def test_stale_or_unreliable_not_published():
    pub = ExternalNavPublisher(min_good_frames=1)
    assert pub.payload(row(sigma_m=100), attitude()) is None
    assert pub.payload(row(ve_mps=90), attitude()) is None
    assert pub.payload(row(1.0), attitude()) is not None
    assert pub.payload(row(1.0), attitude()) is None
    assert pub.payload(row(2.0), attitude()) is None
    assert pub.payload(row(2.2), attitude()) is not None
