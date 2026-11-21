import math
import os
import time

import cv2
import yaml
from pymavlink import mavutil

from navmod.camera import render_frame
from navmod.georef import MapGeoref, latlon_to_px
from navmod.matcher import estimate_pose
from navmod.types import State, GpsMeasurement, ModuleOutput, ModuleStatus


FRAME_SIZE = (512, 512)
GPS_INPUT_GPS_ID = 1  # GPS2: GPS1 залишаємо для штатного GPS SITL.


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


def compute_scale(alt_m, fov_deg, gsd_x, frame_width):
    """Compute render_frame scale from altitude and camera FOV."""
    fov_rad = math.radians(fov_deg)
    ground_width_m = 2 * alt_m * math.tan(fov_rad / 2)
    base_width_m = frame_width * gsd_x
    return ground_width_m / base_width_m


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
    """Copy GPS1 position into GPS2 while GPS1 is available.

    SITL-only bootstrap: GPS2 must have a valid fix before arming.
    This is NOT an independent GPS measurement.
    """

    # GPS_RAW_INT.h_acc is measured in millimetres.
    # If accuracy is unavailable (0), use a test-only fallback.
    horizontal_accuracy = (
        gps_raw.h_acc / 1000.0
        if gps_raw.h_acc > 0
        else 5.0
    )

    output = ModuleOutput(
        lat=gps.lat,
        lon=gps.lon,
        alt_m=gps.alt_m,
        heading_rad=0.0,
        sigma_m=horizontal_accuracy,
        module_status=ModuleStatus.FIX,
    )

    encoded = encode_output(output)

    # GPS1 does not provide a reliable yaw while stationary.
    # GPS_INPUT yaw=0 means "yaw unavailable".
    encoded["yaw"] = 0

    # These values come from the original GPS1 measurement.
    encoded["fix_type"] = 3
    encoded["satellites_visible"] = gps_raw.satellites_visible

    connection.mav.gps_input_send(**encoded)



def request_sim_state(connection):
    """Request SIM_STATE (message 108) at 2 Hz."""
    connection.mav.command_long_send(
        connection.target_system,
        connection.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0,
        108,
        500_000,
        0, 0, 0, 0, 0,
    )


def estimate_visual_fix(state, map_rgba, map_bgr, georef, fov_deg):
    """Generate a synthetic camera frame and match it against the map."""
    px_x, px_y = latlon_to_px(georef, state.lat, state.lon)
    heading_deg = math.degrees(state.yaw_rad)
    scale = compute_scale(state.alt_m, fov_deg, georef.gsd_x, FRAME_SIZE[0])

    frame = render_frame(
        map_bgr, px_x, px_y, heading_deg, scale, FRAME_SIZE, augment=True
    )
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    return estimate_pose(frame, map_bgr, georef)


def main():
    # ============================================================
    # 1. INITIALIZATION — підключення до SITL та завантаження карти
    # ============================================================
    connection = mavutil.mavlink_connection(
        "udpin:127.0.0.1:14551",
        source_system=245,
        dialect="ardupilotmega",
    )

    print("Waiting for SITL...")
    heartbeat = connection.wait_heartbeat(timeout=15)
    if heartbeat is None:
        raise RuntimeError("No heartbeat from SITL")

    print("Connected to ArduPlane!")
    request_sim_state(connection)

    print("Loading map and georef...")
    map_rgba, map_bgr, georef, fov_deg = load_pipeline()
    print(
        f"Map loaded: {map_rgba.shape[1]}x{map_rgba.shape[0]}, "
        f"GSD={georef.gsd_x:.2f} m/px"
    )

    # ============================================================
    # 2. STATE STORAGE — кеш телеметрії та поточне джерело позиції
    # ============================================================
    messages = {}
    last_received = {"HEARTBEAT": time.monotonic()}

    link_alive = True
    heartbeat_timeout = 3.0

    # Діагностичне джерело: GPS, MATCHER або NO_FIX.
    nav_source = None

    # Окремо обмежуємо СПРОБИ matching і успішні GPS_INPUT.
    # Так навіть при fix.ok=False ми не запускаємо matcher безперервно.
    last_match_attempt = 0.0
    last_sent = 0.0
    match_interval = 0.2  # Максимум 5 спроб matching за секунду.
    last_relay_sent = 0.0
    relay_interval = 0.1  # GPS1 -> GPS2, максимум 10 Hz

    # ============================================================
    # 3. MAIN LOOP — повідомлення MAVLink і вибір джерела позиції
    # ============================================================
    while True:
        msg = connection.recv_match(
            type=[
                "HEARTBEAT",
                "SIM_STATE",
                "SIMSTATE",
                "ATTITUDE",
                "GLOBAL_POSITION_INT",
                "GPS_RAW_INT",
            ],
            blocking=True,
            timeout=0.05,
        )
        now = time.monotonic()

        # --------------------------------------------------------
        # 3.1. Перевірка heartbeat та відновлення після перезапуску
        # --------------------------------------------------------
        if (
            msg is not None
            and msg.get_type() == "HEARTBEAT"
            and msg.autopilot == mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA
        ):
            if not link_alive:
                last_relay_sent = 0.0
                print("SITL connection restored!")
                connection.target_system = msg.get_srcSystem()
                connection.target_component = msg.get_srcComponent()
                messages.clear()
                last_received.clear()
                last_match_attempt = 0.0
                last_sent = 0.0
                nav_source = None
                request_sim_state(connection)

            last_received["HEARTBEAT"] = now
            link_alive = True

        if now - last_received["HEARTBEAT"] > heartbeat_timeout:
            if link_alive:
                print("WARNING: SITL connection lost!")
                messages.clear()
                nav_source = None
            link_alive = False

        # ============================================================
        # GPS2 RELAY — незалежна періодична передача GPS_INPUT
        # ============================================================

        if link_alive and now - last_relay_sent >= relay_interval:

            gps = parse_gps(messages, last_received)

            if gps.available:
                gps_raw = messages["GPS_RAW_INT"]

                relay_gps1_to_gps2(
                    connection,
                    gps,
                    gps_raw,
                )

                last_relay_sent = now

        if not link_alive or msg is None:
            continue

        # --------------------------------------------------------
        # 3.2. Зберігання останнього повідомлення кожного типу
        # --------------------------------------------------------
        msg_type = msg.get_type()
        messages[msg_type] = msg
        if msg_type != "HEARTBEAT":
            last_received[msg_type] = now

        if msg_type == "GPS_RAW_INT":
            gps = parse_gps(messages, last_received)
            print(
                f"GPS1: available={gps.available}, "
                f"lat={gps.lat:.7f}, lon={gps.lon:.7f}, "
                f"alt={gps.alt_m:.2f}"
            )
            continue

        if msg_type == "SIM_STATE":
            print("TRUE SIM ALT:", msg.alt)
            continue

        # Увесь optical pipeline запускається тільки на новий SIMSTATE.
        if msg_type != "SIMSTATE":
            continue

        # --------------------------------------------------------
        # 3.3. Найвищий пріоритет: штатний GPS1
        # --------------------------------------------------------

        # GPS1 has the highest navigation priority.
        gps = parse_gps(messages, last_received)

        if gps.available:

            # Keep GPS2 initialized and healthy before and during flight.
            # GPS2 receives GPS1 coordinates while the original GPS works.
            if now - last_relay_sent >= relay_interval:
                gps_raw = messages["GPS_RAW_INT"]

                relay_gps1_to_gps2(
                    connection,
                    gps,
                    gps_raw,
                )

                last_relay_sent = now

            if nav_source != "GPS":
                print("NAV SOURCE -> GPS (GPS2 relay active)")
                nav_source = "GPS"

            continue

        # GPS1 unavailable:
        # Continue into the existing visual matching pipeline below.


        # --------------------------------------------------------
        # 3.4. GPS немає: пробуємо visual matching
        # --------------------------------------------------------
        if now - last_match_attempt < match_interval:
            continue
        last_match_attempt = now

        state = parse_state(messages, last_received)
        sim_state = messages.get("SIM_STATE")
        sim_time = last_received.get("SIM_STATE")

        # Потрібні свіжі дані та додатна висота для масштабу кадру.
        if (
            state is None
            or sim_state is None
            or sim_time is None
            or now - sim_time > 2.0
            or state.alt_m <= 0.0
        ):
            if nav_source != "NO_FIX":
                print("NAV SOURCE -> NO_FIX (missing/stale simulator data)")
                nav_source = "NO_FIX"
            # TODO: visual odometry — оцінка переміщення між кадрами.
            continue

        # Функція повертає Fix(..., sigma_m, ok).
        fix = estimate_visual_fix(state, map_rgba, map_bgr, georef, fov_deg)

        # --------------------------------------------------------
        # 3.5. Рішення matcher: перевіряємо ТІЛЬКИ fix.ok
        # --------------------------------------------------------
        # sigma_m не впливає на перемикання режимів. Її передаємо
        # в ArduPilot як horiz_accuracy, без додаткового порога.
        if not fix.ok:
            if nav_source != "NO_FIX":
                print("NAV SOURCE -> NO_FIX (matcher failed)")
                nav_source = "NO_FIX"
            # TODO: visual odometry — fallback, коли fix.ok=False.
            continue

        # --------------------------------------------------------
        # 3.6. Успішний matcher: надсилаємо координати в GPS2
        # --------------------------------------------------------
        output = ModuleOutput(
            lat=fix.lat,
            lon=fix.lon,
            alt_m=sim_state.alt,
            heading_rad=fix.heading_rad,
            sigma_m=fix.sigma_m,
            module_status=ModuleStatus.FIX,
        )
        encoded = encode_output(output)
        connection.mav.gps_input_send(**encoded)

        if nav_source != "MATCHER":
            print("NAV SOURCE -> MATCHER")
            nav_source = "MATCHER"

        sent_at = time.monotonic()
        if last_sent > 0:
            frequency = 1.0 / (sent_at - last_sent)
            print(f"GPS_INPUT frequency: {frequency:.2f} Hz")
        last_sent = sent_at

        print(
            "MATCHER GPS_INPUT:",
            encoded["lat"],
            encoded["lon"],
            "alt:", encoded["alt"],
            "sigma:", f"{fix.sigma_m:.2f}m",
        )


if __name__ == "__main__":
    main()
