"""SITL navigation bridge with a per-run 5 Hz diagnostic CSV."""

import math
import os
import time
import sys
import csv
import json
import socket
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import yaml
from pymavlink import mavutil

from navmod.camera import render_frame
from navmod.georef import MapGeoref, latlon_to_px
from navmod.matcher import estimate_pose
from navmod.odometry import camera_gsd
from navmod.vo_diagnostics import VODiagnosticRunner
from navmod.vo_gps2 import VOGps2Adapter
from navmod.vo_mavlink import ExternalNavPublisher
from navmod.matcher_vo_fusion import MatcherFirstFusion
from navmod.types import (
    State, GpsMeasurement, ModuleOutput, ModuleStatus, Fix, CameraFrame,
)

# src/logger.py is a standalone module, not a submodule of navmod.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from logger import RunLogger


FRAME_SIZE = (512, 512)
GPS_INPUT_GPS_ID = 1  # GPS2: GPS1 залишаємо для штатного GPS SITL.
GPS2_SOURCE = os.getenv("NAVMOD_GPS2_SOURCE", "matcher").lower()
if GPS2_SOURCE not in {"matcher", "vo"}:
    raise ValueError("NAVMOD_GPS2_SOURCE must be matcher or vo")
NAVMOD_NAV_OUTPUT = os.getenv("NAVMOD_NAV_OUTPUT", "gps2").lower()
if NAVMOD_NAV_OUTPUT not in {"gps2", "externalnav"}:
    raise ValueError("NAVMOD_NAV_OUTPUT must be gps2 or externalnav")
EXTERNALNAV_MODE = NAVMOD_NAV_OUTPUT == "externalnav"
# Matcher runs separately from the 5 Hz VO loop.  A late match is corrected
# to the frame on which it was computed (using a VO snapshot).
MATCHER_RATE_HZ = max(0.2, float(os.getenv("NAVMOD_MATCH_HZ", "1.5")))
# Diagnostic-only UDP. Never enters ArduPilot or changes EKF sources.
MATCHER_MAP_PORT = int(os.getenv("NAVMOD_MATCHER_MAP_PORT", "15379"))


def load_pipeline(config_path="config/default.yaml"):
    """Load map, georef, and camera config for the optical pipeline."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    with open(os.path.join(repo_root, config_path)) as f:
        cfg = yaml.safe_load(f)

    map_path = os.path.join(repo_root, cfg["map"]["path"])
    map_rgba = cv2.imread(map_path, cv2.IMREAD_UNCHANGED)
    if map_rgba is None:
        raise FileNotFoundError(f"Map not found: {map_path}")

    h, w = map_rgba.shape[:2]
    georef = MapGeoref(
        top_left_lat=cfg["map"]["top_left_lat"],
        top_left_lon=cfg["map"]["top_left_lon"],
        bottom_right_lat=cfg["map"]["bottom_right_lat"],
        bottom_right_lon=cfg["map"]["bottom_right_lon"],
        image_width_px=w,
        image_height_px=h,
    )

    if map_rgba.shape[2] == 4:
        map_bgr = cv2.cvtColor(map_rgba, cv2.COLOR_BGRA2BGR)
    else:
        map_bgr = map_rgba

    fov_deg = cfg["camera"]["fov_deg"]
    return map_rgba, map_bgr, georef, fov_deg



def compute_scale(
    alt_m,
    fov_deg,
    gsd_x,
    frame_width,
    frame_height,
):
    gsd = camera_gsd(
        alt_m,
        fov_deg,
        frame_width,
        frame_height,
    )

    return gsd / gsd_x



def parse_state(messages, last_received) -> State | None:
    """Convert fresh SITL truth and altitude into the aircraft State.

    SIMSTATE contains true lat/lon/attitude. GLOBAL_POSITION_INT is used
    only for relative altitude and timestamp (its lat/lon may be zero).
    """
    required = ("SIMSTATE", "GLOBAL_POSITION_INT")
    now = time.monotonic()

    for msg_type in required:
        if msg_type not in messages or msg_type not in last_received:
            return None
        if now - last_received[msg_type] > 2.0:
            return None

    sim = messages["SIMSTATE"]
    pos = messages["GLOBAL_POSITION_INT"]

    return State(
        lat=sim.lat / 1e7,
        lon=sim.lng / 1e7,
        alt_m=pos.relative_alt / 1000.0,
        roll_rad=sim.roll,
        pitch_rad=sim.pitch,
        yaw_rad=sim.yaw,
        timestamp_s=pos.time_boot_ms / 1000.0,
    )


def parse_gps(messages, last_received) -> GpsMeasurement:
    """Check ONLY GPS1's GPS_RAW_INT for a fresh 3D fix."""
    gps = messages.get("GPS_RAW_INT")
    gps_time = last_received.get("GPS_RAW_INT")

    if gps is None or gps_time is None:
        return GpsMeasurement(available=False)

    if time.monotonic() - gps_time > 2.0:
        return GpsMeasurement(available=False)

    if gps.fix_type < 3:
        return GpsMeasurement(available=False)

    return GpsMeasurement(
        lat=gps.lat / 1e7,
        lon=gps.lon / 1e7,
        alt_m=gps.alt / 1000.0,
        speed_mps=gps.vel / 100.0 if gps.vel != 65535 else 0.0,
        available=True,
    )


def encode_output(output: ModuleOutput) -> dict:
    """Encode an optical position fix as GPS_INPUT for the second GPS."""
    if output.module_status != ModuleStatus.FIX:
        raise ValueError("Cannot encode GPS_INPUT: no reliable position fix")

    heading_deg = math.degrees(output.heading_rad) % 360
    yaw = round(heading_deg * 100)
    if yaw == 0 or yaw == 36000:
        yaw = 36000  # MAVLink uses 36000 for true north.

    time_usec = time.time_ns() // 1000
    gps_epoch_usec = 315964800 * 1_000_000
    gps_elapsed_ms = (time_usec - gps_epoch_usec) // 1000 + 18_000
    gps_week, time_week_ms = divmod(gps_elapsed_ms, 604_800_000)

    return {
        "lat": round(output.lat * 1e7),
        "lon": round(output.lon * 1e7),
        "yaw": yaw,
        "horiz_accuracy": output.sigma_m,
        "alt": output.alt_m,
        "gps_id": GPS_INPUT_GPS_ID,
        "fix_type": 3,
        "ignore_flags": 190,
        "time_usec": time_usec,
        "time_week": gps_week,
        "time_week_ms": time_week_ms,
        "hdop": 0.0,
        "vdop": 0.0,
        "vn": 0.0,
        "ve": 0.0,
        "vd": 0.0,
        "speed_accuracy": 0.0,
        "vert_accuracy": 0.0,
        "satellites_visible": 0,
    }


def relay_gps1_to_gps2(connection, gps, gps_raw):
    """SITL-only GPS2 bootstrap: this is NOT an independent GPS sensor."""
    accuracy_m = gps_raw.h_acc / 1000.0 if gps_raw.h_acc > 0 else 5.0
    output = ModuleOutput(
        lat=gps.lat, lon=gps.lon, alt_m=gps.alt_m,
        heading_rad=0.0, sigma_m=accuracy_m, module_status=ModuleStatus.FIX,
    )
    payload = encode_output(output)
    payload["yaw"] = 0  # Yaw unavailable for a stationary GPS.
    payload["satellites_visible"] = gps_raw.satellites_visible
    connection.mav.gps_input_send(**payload)


def request_sim_state(connection):
    """Request 5 Hz truth / clock telemetry; firmware can limit stream rates."""
    for msg_id, interval_us in (
        (108, 200_000),  # SIM_STATE: true MSL altitude
        (164, 200_000),  # SIMSTATE: true latitude, longitude and attitude
        (30, 100_000),   # ATTITUDE: has time_boot_ms for .bin time alignment
        (2, 200_000),    # SYSTEM_TIME: backup time_boot_ms
    ):
        connection.mav.command_long_send(
            connection.target_system,
            connection.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0, msg_id, interval_us, 0, 0, 0, 0, 0,
        )
    # Query the true home MSL altitude for approximately ground-relative height.
    connection.mav.command_long_send(
        connection.target_system,
        connection.target_component,
        mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE,
        0, 242, 0, 0, 0, 0, 0, 0,  # HOME_POSITION
    )


def generate_camera_frame(state, map_bgr, georef, fov_deg):
    """Generate one synthetic camera frame from SITL truth."""

    px_x, px_y = latlon_to_px(
        georef, state.lat, state.lon
    )

    heading_deg = math.degrees(state.yaw_rad)

    scale = compute_scale(
        state.alt_m,
        fov_deg,
        georef.gsd_x,
        FRAME_SIZE[0],
        FRAME_SIZE[1],
    )

    frame = render_frame(
        map_bgr,
        px_x,
        px_y,
        heading_deg,
        scale,
        FRAME_SIZE,
        augment=True,
    )

    if frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    return frame


def _fresh(messages, received, name, now, max_age=1.0):
    """Return a recent telemetry message or None (avoid invented ground truth)."""
    if name not in messages or now - received.get(name, float("-inf")) > max_age:
        return None
    return messages[name]


def _boot_seconds(messages, received, now):
    """Autopilot boot clock, compatible with DataFlash TimeUS / 1e6.

    Neither HEARTBEAT nor SIMSTATE has time_boot_ms. Use the newest
    boot timestamp from ATTITUDE / GLOBAL_POSITION_INT / SYSTEM_TIME.
    """
    observations = []
    for name in ("ATTITUDE", "GLOBAL_POSITION_INT", "SYSTEM_TIME"):
        packet = _fresh(messages, received, name, now, max_age=1.0)
        if packet is not None and hasattr(packet, "time_boot_ms"):
            observations.append((received[name], packet.time_boot_ms))
    if not observations:
        return None
    return max(observations, key=lambda pair: pair[0])[1] / 1000.0


def _truth_and_camera_state(messages, received, now, home_alt_amsl, t_sim):
    """Use SITL truth for location/heading and true MSL altitude if available.

    Ground height is approximated by home altitude (flat-ground assumption).
    If HOME_POSITION is missing, gt_alt remains empty rather than using EKF.
    The old EKF relative_alt is used only as a camera-height fallback.
    """
    sim = _fresh(messages, received, "SIMSTATE", now, max_age=0.7)
    if sim is None:
        return None, None

    sim_alt = _fresh(messages, received, "SIM_STATE", now, max_age=0.7)
    agl = (sim_alt.alt - home_alt_amsl
           if sim_alt is not None and home_alt_amsl is not None else None)

    gt = {
        "lat": sim.lat / 1e7,
        "lon": sim.lng / 1e7,
        "yaw_rad": sim.yaw,
        "alt_m": agl,
    }

    camera_alt = agl
    if camera_alt is None or camera_alt <= 0:
        pos = _fresh(messages, received, "GLOBAL_POSITION_INT", now)
        camera_alt = pos.relative_alt / 1000.0 if pos is not None else None
    if camera_alt is None or camera_alt <= 0:
        return gt, None

    camera_state = State(
        lat=gt["lat"], lon=gt["lon"], alt_m=camera_alt,
        roll_rad=sim.roll, pitch_rad=sim.pitch, yaw_rad=sim.yaw,
        timestamp_s=t_sim,
    )
    return gt, camera_state


def main():
    # A logger is instantiated once per process run, before connecting to SITL.
    with RunLogger("logs/") as logger:
        print(f"CSV run log: {logger.path}")
        _run_bridge(logger)


def _run_bridge(logger):
    connection = mavutil.mavlink_connection(
        "udpin:127.0.0.1:14551", source_system=245, dialect="ardupilotmega",
    )
    print("Waiting for SITL...")
    heartbeat = connection.wait_heartbeat(timeout=15)
    if heartbeat is None:
        raise RuntimeError("No heartbeat from SITL")
    print("Connected to ArduPlane!")
    request_sim_state(connection)

    map_rgba, map_bgr, georef, fov_deg = load_pipeline()
    print(f"Map loaded: {map_bgr.shape[1]}x{map_bgr.shape[0]}, GSD={georef.gsd_x:.2f} m/px")

    # Diagnostic sidecar: no data from VO goes to the autopilot.
    vo_diag = VODiagnosticRunner(
        logger.path.with_name(logger.path.stem + "_vo.csv"),
        georef=georef,
        fov_diag_deg=fov_deg,
    )
    print(f"VO diagnostic log: {vo_diag.path}")
    vo_gps = VOGps2Adapter(georef)
    external_pub = ExternalNavPublisher(min_good_frames=12)
    fusion = MatcherFirstFusion(georef)
    match_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="map-matcher") if EXTERNALNAV_MODE else None
    match_future = None
    match_snapshot = None
    last_match_submit = float("-inf")
    # Send accepted raw matcher fixes to a local MAVProxy map plugin.
    # This socket NEVER sends GPS_INPUT or ODOMETRY to the vehicle.
    matcher_map_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if EXTERNALNAV_MODE else None
    matcher_map_packets = 0
    match_log = None
    match_writer = None
    if EXTERNALNAV_MODE:
        match_path = logger.path.with_name(logger.path.stem + "_matcher_async.csv")
        match_log = match_path.open("x", newline="", encoding="utf-8")
        match_writer = csv.DictWriter(match_log, fieldnames=(
            "frame_id", "t_capture_s", "t_received_s", "age_s",
            "ok", "accepted", "reason", "innovation_m",
            "fix_lat", "fix_lon", "n_inliers", "inlier_ratio",
        ))
        match_writer.writeheader()
        print(f"Async matcher diagnostics: {match_path}")
    if EXTERNALNAV_MODE:
        print("ExternalNav ODOMETRY: enabled (source selection remains MANUAL)")
    print(f"GPS2 optical source: {GPS2_SOURCE.upper()} (SITL test)")
    last_vo_simstate_rx = None

    last_ground_odom_tx = 0.0
    ground_bootstrap_active = False

    messages = {}
    last_received = {"HEARTBEAT": time.monotonic()}
    link_alive = True
    heartbeat_timeout = 3.0
    last_relay_sent = 0.0
    relay_interval = 0.1  # best effort 10 Hz, subject to matcher execution time
    tick_interval = 0.2  # target 5 Hz in SITL boot time
    next_tick_t = None
    frame_id = 0
    nav_source = None
    last_match_sent = 0.0
    last_vo_report = 0.0
    home_alt_amsl = None

    try:
        while True:
            msg = connection.recv_match(
                type=[
                    "HEARTBEAT", "SIM_STATE", "SIMSTATE", "ATTITUDE",
                    "GLOBAL_POSITION_INT", "GPS_RAW_INT", "SYSTEM_TIME",
                    "HOME_POSITION",
                ],
                blocking=True,
                timeout=0.02,
            )
            now = time.monotonic()

            if (msg is not None and msg.get_type() == "HEARTBEAT"
                    and msg.autopilot == mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA):
                if not link_alive:
                    print("SITL connection restored!")
                    messages.clear()
                    last_received.clear()
                    last_relay_sent = 0.0
                    next_tick_t = None
                    home_alt_amsl = None
                    nav_source = None
                    vo_diag.mark_gap()
                    vo_gps.reset()
                    external_pub.reset()
                    fusion.reset()
                    last_vo_simstate_rx = None
                    connection.target_system = msg.get_srcSystem()
                    connection.target_component = msg.get_srcComponent()
                    request_sim_state(connection)
                last_received["HEARTBEAT"] = now
                link_alive = True

            if now - last_received.get("HEARTBEAT", float("-inf")) > heartbeat_timeout:
                if link_alive:
                    print("WARNING: SITL connection lost")
                    messages.clear()
                    vo_diag.mark_gap()
                    vo_gps.reset()
                    external_pub.reset()
                    fusion.reset()
                    last_vo_simstate_rx = None
                link_alive = False
                continue

            if msg is not None:
                msg_type = msg.get_type()
                messages[msg_type] = msg
                last_received[msg_type] = now
                if msg_type == "HOME_POSITION":
                    home_alt_amsl = msg.altitude / 1000.0

                # Fallback home reference ONLY if we actually observe the
                # plane on/near the ground at bridge startup.
                if msg_type == "SIM_STATE" and home_alt_amsl is None:
                    gpi = _fresh(messages, last_received, "GLOBAL_POSITION_INT", now)
                    if gpi is not None and abs(gpi.relative_alt) < 1500:
                        home_alt_amsl = msg.alt

            if not link_alive:
                continue

            gps = parse_gps(messages, last_received)
            if EXTERNALNAV_MODE:
                fusion.seed_origin(gps)

            # SITL-only pre-arm ExternalNav bootstrap.
            # Use a stationary local pose while the aircraft is disarmed.
            if EXTERNALNAV_MODE:
                hb = messages.get("HEARTBEAT")
                armed = (
                    hb is not None
                    and bool(
                        hb.base_mode
                        & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                    )
                )

                att = _fresh(
                    messages, last_received,
                    "ATTITUDE", now, max_age=0.3,
                )

                if (
                    not armed
                    and gps.available
                    and att is not None
                    and now - last_ground_odom_tx >= 0.1
                ):
                    q = external_pub._quaternion(
                        att.roll, att.pitch, att.yaw
                    )
                    
                    # All covariance values must be finite.
                    cov = [0.0] * 21

                    # Position variances, m²
                    cov[0] = 25.0
                    cov[6] = 25.0
                    cov[11] = 100.0

                    # Angular variances, rad²
                    cov[15] = math.radians(15.0) ** 2
                    cov[18] = math.radians(15.0) ** 2
                    cov[20] = math.radians(30.0) ** 2

                    # Velocity variances
                    vel_cov = [0.0] * 21
                    vel_cov[0] = 4.0
                    vel_cov[6] = 4.0
                    vel_cov[11] = 4.0

                    connection.mav.odometry_send(
                        time_usec=time.monotonic_ns() // 1000,
                        frame_id=20,
                        child_frame_id=12,
                        x=0.0, y=0.0, z=0.0,
                        q=list(q),
                        vx=0.0, vy=0.0, vz=0.0,
                        rollspeed=0.0,
                        pitchspeed=0.0,
                        yawspeed=0.0,
                        pose_covariance=cov,
                        velocity_covariance=vel_cov,
                        reset_counter=external_pub.reset_counter,
                        estimator_type=0,
                        quality=0,
                    )

                    last_ground_odom_tx = now
                    ground_bootstrap_active = True

                elif armed and ground_bootstrap_active:
                    external_pub.reset()
                    ground_bootstrap_active = False

            if (not EXTERNALNAV_MODE and gps.available
                    and now - last_relay_sent >= relay_interval):
                # GPS1->GPS2 bootstrap (SITL only), never counted as matcher sent_to_ekf.
                try:
                    relay_gps1_to_gps2(connection, gps, messages["GPS_RAW_INT"])
                    last_relay_sent = now
                except Exception as exc:
                    print(f"GPS2 relay send failed: {exc}")

            t_sim = _boot_seconds(messages, last_received, now)
            if t_sim is None:
                continue
            if next_tick_t is not None and t_sim < next_tick_t:
                # On simulation reboot, t_sim may reset: start a new schedule.
                if t_sim + 1.0 < next_tick_t - tick_interval:
                    next_tick_t = None
                else:
                    continue

            # Each loop below corresponds to ONE logged navigation tick.
            # No fabricated catch-up rows if matching takes >200 ms.
            next_tick_t = t_sim + tick_interval
            gt, state = _truth_and_camera_state(
                messages, last_received, now, home_alt_amsl, t_sim,
            )
            gps_state = "ok" if gps.available else "off"
            fix = Fix(
                lat=float("nan"), lon=float("nan"),
                heading_rad=float("nan"), sigma_m=float("inf"), ok=False,
            )
            sent_to_ekf = False
            optical_tx_source = ""
            vo_row = None
            attitude = None

            # Collect completed asynchronous map match; associate it with the
            # image capture timestamp, not with this loop's current frame.
            if EXTERNALNAV_MODE and match_future is not None and match_future.done():
                try:
                    fix = match_future.result()
                    decision = fusion.accept_match(fix, match_snapshot, t_sim)
                except Exception as exc:
                    decision = None
                    print(f"Async matcher error: {exc}")
                if decision is not None:
                    match_writer.writerow({
                        "frame_id": match_snapshot.frame_id,
                        "t_capture_s": match_snapshot.t_sim,
                        "t_received_s": t_sim,
                        "age_s": t_sim - match_snapshot.t_sim,
                        "ok": bool(fix.ok),
                        "accepted": decision.accepted,
                        "reason": decision.reason,
                        "innovation_m": decision.innovation_m if decision.innovation_m is not None else "",
                        "fix_lat": fix.lat if fix.ok else "",
                        "fix_lon": fix.lon if fix.ok else "",
                        "n_inliers": fix.n_inliers,
                        "inlier_ratio": fix.inlier_ratio,
                    })
                    match_log.flush()
                    if decision.accepted:
                        print(f"MATCHER fix accepted frame={match_snapshot.frame_id} "
                              f"age={t_sim-match_snapshot.t_sim:.2f}s "
                              f"innovation={decision.innovation_m}")
                        # Independent visual marker for the RAW matcher position.
                        # Do not repurpose GPS2: the EKF remains on ExternalNav.
                        if matcher_map_socket is not None and fix.ok:
                            try:
                                heading = math.degrees(float(fix.heading_rad))
                                if not math.isfinite(heading):
                                    heading = 0.0
                                marker = {
                                    "type": "matcher_fix",
                                    "lat": float(fix.lat),
                                    "lon": float(fix.lon),
                                    "heading_deg": heading % 360.0,
                                    "frame_id": int(match_snapshot.frame_id),
                                    "age_s": max(0.0, t_sim - match_snapshot.t_sim),
                                }
                                if all(math.isfinite(marker[k]) for k in
                                       ("lat", "lon", "heading_deg", "age_s")):
                                    matcher_map_socket.sendto(
                                        json.dumps(marker).encode("utf-8"),
                                        ("127.0.0.1", MATCHER_MAP_PORT),
                                    )
                                    matcher_map_packets += 1
                            except (OSError, TypeError, ValueError) as map_exc:
                                print(f"Matcher map telemetry warning: {map_exc}")
                match_future = None
                match_snapshot = None

            # VO runs on fresh SIMSTATE frames.  Matcher independently runs
            # asynchronously on the same camera frames (including with GPS1).
            if state is not None:
                try:
                    image = generate_camera_frame(
                        state, map_bgr, georef, fov_deg
                    )

                    camera_frame = CameraFrame(
                        image=image,
                        timestamp_s=state.timestamp_s,
                        height_agl_m=state.alt_m,
                        frame_id=frame_id,
                    )

                    # Read-only VO: use each new SIMSTATE packet once.
                    # Passing repeated truth frames with independent noise creates
                    # fake optical flow even when the aircraft does not move.
                    if state.alt_m > 5.0:
                        simstate_rx = last_received.get("SIMSTATE")
                        if (simstate_rx is not None
                                and simstate_rx != last_vo_simstate_rx):
                            last_vo_simstate_rx = simstate_rx
                            attitude = _fresh(
                                messages, last_received, "ATTITUDE", now,
                                max_age=0.3,
                            )
                            if attitude is None:
                                vo_diag.mark_gap()
                            else:
                                try:
                                    # The estimator sees a frame + AHRS yaw.
                                    # True lat/lon is used ONLY by the CSV scorer.
                                    vo_row = vo_diag.observe(
                                        camera_frame,
                                        yaw_rad=(attitude.yaw if EXTERNALNAV_MODE
                                                 else gt["yaw_rad"]),
                                        gt_lat=gt["lat"],
                                        gt_lon=gt["lon"],
                                        gt_yaw_rad=gt["yaw_rad"],
                                        simstate_age_ms=(
                                            now - simstate_rx
                                        ) * 1000.0,
                                        attitude_age_ms=(
                                            now - last_received["ATTITUDE"]
                                        ) * 1000.0,
                                    )
                                except Exception as vo_exc:
                                    print(f"VO diagnostic error: {vo_exc}")
                                    vo_diag.mark_gap()

                    if EXTERNALNAV_MODE:
                        # A fast VO tick never waits for ORB matching.
                        # Schedule only when this is a new, VO-processed image;
                        # keep at most ONE matcher job in flight.
                        snap = fusion.observe_frame(vo_row, gps)
                        if (snap is not None
                                and match_future is None
                                and now - last_match_submit >= 1.0 / MATCHER_RATE_HZ):
                            match_snapshot = snap
                            match_future = match_executor.submit(
                                estimate_pose, camera_frame.image.copy(), map_bgr, georef
                            )
                            last_match_submit = now
                    else:
                        fix = estimate_pose(camera_frame.image, map_bgr, georef)

                except Exception as exc:
                    print(f"Visual pipeline error: {exc}")

            vo_fix = (vo_gps.update(vo_row=vo_row, gps=gps, matcher_fix=fix)
                      if GPS2_SOURCE == "vo" and not EXTERNALNAV_MODE else None)

            if EXTERNALNAV_MODE:
                # This only transmits ExternalNav. The user must switch
                # EKF source set separately after verifying health.
                payload = external_pub.payload(vo_row, attitude)
                fused = fusion.fused_ned(vo_row) if payload is not None else None
                if payload is not None and fused is not None:
                    fused_n, fused_e, fused_source, correction_residual = fused
                    # Keep ONE consistent local NED frame; matcher corrects the
                    # absolute drift, VO propagates every new valid image.
                    payload["x"] = fused_n
                    payload["y"] = fused_e
                    # Never transmit NaN covariance entries to ArduPilot.
                    velcov = [0.0] * 21
                    velcov[0] = velcov[6] = 9.0   # (3 m/s)^2, provisional
                    velcov[11] = 25.0            # vertical speed not estimated
                    payload["velocity_covariance"] = velcov
                    # During a significant matcher correction, do not claim
                    # high precision to EKF.
                    if correction_residual > 1.0:
                        posecov = list(payload["pose_covariance"])
                        var = max(posecov[0], min(correction_residual, 40.0)**2)
                        posecov[0] = posecov[6] = var
                        payload["pose_covariance"] = posecov
                    try:
                        connection.mav.odometry_send(**payload)
                    except Exception as exc:
                        print(f"ExternalNav ODOMETRY send failed: {exc}")
                    else:
                        sent_to_ekf = True  # sent, NOT proof of EKF fusion
                        optical_tx_source = fused_source
                        if now - last_vo_report > 2.0:
                            print(f"FUSED ODOMETRY: source={fused_source} "
                                  f"segment={vo_row['segment_id']} "
                                  f"correction={correction_residual:.1f}m "
                                  f"reset={payload['reset_counter']}")
                            last_vo_report = now
            elif gps.available:
                if nav_source != "GPS":
                    nav_source = "GPS"
                    print("NAV SOURCE -> GPS (GPS2 relay active)")
            elif GPS2_SOURCE == "vo":
                # SITL-only GPS2 experiment. No truth lat/lon enters VO.
                altitude = _fresh(messages, last_received, "SIM_STATE", now)
                if vo_fix is not None and altitude is not None:
                    output = ModuleOutput(
                        lat=vo_fix.lat, lon=vo_fix.lon, alt_m=altitude.alt,
                        heading_rad=0.0, sigma_m=vo_fix.reported_accuracy_m,
                        module_status=ModuleStatus.FIX,
                    )
                    payload = encode_output(output)
                    payload["yaw"] = 0  # Unknown: VO does not estimate absolute yaw.
                    try:
                        connection.mav.gps_input_send(**payload)
                    except Exception as exc:
                        print(f"VO GPS_INPUT send failed: {exc}")
                    else:
                        sent_to_ekf = True
                        optical_tx_source = "vo"
                        if nav_source != "VO":
                            nav_source = "VO"
                            print(f"NAV SOURCE -> VO (anchor={vo_fix.anchor_source})")
                        if now - last_vo_report > 2.0:
                            print(f"VO GPS2 sent: sigma={vo_fix.vo_sigma_m:.1f}m "
                                  f"reported_accuracy={vo_fix.reported_accuracy_m:.1f}m "
                                  f"segment={vo_fix.segment_id}")
                            last_vo_report = now
                elif nav_source != "VO_NO_FIX":
                    nav_source = "VO_NO_FIX"
                    print("NAV SOURCE -> VO_NO_FIX (waiting for anchor/track)")
            elif fix.ok and state is not None:
                altitude = _fresh(messages, last_received, "SIM_STATE", now)
                if altitude is not None:
                    output = ModuleOutput(
                        lat=fix.lat, lon=fix.lon, alt_m=altitude.alt,
                        heading_rad=fix.heading_rad, sigma_m=fix.sigma_m,
                        module_status=ModuleStatus.FIX,
                    )
                    try:
                        connection.mav.gps_input_send(**encode_output(output))
                    except Exception as exc:
                        print(f"Matcher GPS_INPUT send failed: {exc}")
                    else:
                        sent_to_ekf = True
                        optical_tx_source = "matcher"
                        if nav_source != "MATCHER":
                            nav_source = "MATCHER"
                            print("NAV SOURCE -> MATCHER")
                        now_sent = time.monotonic()
                        if last_match_sent:
                            print(f"Matcher GPS_INPUT Hz: {1/(now_sent-last_match_sent):.2f}")
                        last_match_sent = now_sent
            else:
                if nav_source != "NO_FIX":
                    nav_source = "NO_FIX"
                    print("NAV SOURCE -> NO_FIX")

            # Exactly once, AFTER the GPS_INPUT attempt, even for fix.ok=False.
            logger.log_tick(
                t_sim=t_sim, frame_id=frame_id, gt=gt, fix=fix,
                branch_won="",  # No texture/structure/fused branches in main.
                sent_to_ekf=sent_to_ekf, gps_state=gps_state,
                optical_tx_source=optical_tx_source,
            )
            frame_id += 1
    finally:
        vo_diag.close()
        if match_log is not None:
            match_log.flush()
            match_log.close()
        if match_executor is not None:
            match_executor.shutdown(wait=False, cancel_futures=True)
        if matcher_map_socket is not None:
            matcher_map_socket.close()
        # The with-block in main() closes and flushes CSV, even on Ctrl+C.
        connection.close()


if __name__ == "__main__":
    main()
