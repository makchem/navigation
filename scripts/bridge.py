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
    """Convert fresh MAVLink telemetry into aircraft State."""

    required = ["SIMSTATE", "GLOBAL_POSITION_INT"]
    now = time.monotonic()

    for msg_type in required:
        if msg_type not in messages:
            return None

        if msg_type not in last_received:
            return None

        age = now - last_received[msg_type]

        if age > 2.0:
            return None

    sim = messages["SIMSTATE"]
    pos = messages["GLOBAL_POSITION_INT"]

    print(
    "GLOBAL ALT MSL:", pos.alt / 1000.0,
    "RELATIVE ALT:", pos.relative_alt / 1000.0,
)

    return State(
        lat=sim.lat / 1e7,
        lon=sim.lng / 1e7,
        alt_m=pos.relative_alt / 1000.0,
        roll_rad=sim.roll,
        pitch_rad=sim.pitch,
        yaw_rad=sim.yaw,
        timestamp_s=pos.time_boot_ms / 1000.0
    )




def parse_gps(messages, last_received) -> GpsMeasurement:
    """Convert fresh MAVLink GPS telemetry into GPS measurement."""

    gps = messages.get("GPS_RAW_INT")
    gps_time = last_received.get("GPS_RAW_INT")

    if gps is None or gps_time is None:
        return GpsMeasurement(available=False)

    age = time.monotonic() - gps_time

    if age > 2.0:
        return GpsMeasurement(available=False)

    if gps.fix_type < 3:
        return GpsMeasurement(available=False)

    return GpsMeasurement(
        lat=gps.lat / 1e7,
        lon=gps.lon / 1e7,
        alt_m=gps.alt / 1000.0,
        speed_mps=gps.vel / 100.0 if gps.vel != 65535 else 0.0,
        available=True
    )





def encode_output(output: ModuleOutput) -> dict:
    """Prepare position fields for MAVLink GPS_INPUT."""

    if output.module_status != ModuleStatus.FIX:
        raise ValueError(
            "Cannot encode GPS_INPUT: no reliable position fix"
        )

    heading_deg = math.degrees(output.heading_rad) % 360

    yaw = round(heading_deg * 100)

    # MAVLink uses 36000 to represent true north.
    if yaw == 0 or yaw == 36000:
        yaw = 36000

    
    time_usec = time.time_ns() // 1000

    # Unix timestamp of GPS epoch: 1980-01-06
    gps_epoch_usec = 315964800 * 1_000_000

    # GPS time is currently 18 seconds ahead of UTC
    gps_elapsed_ms = (time_usec - gps_epoch_usec) // 1000 + 18_000

    gps_week, time_week_ms = divmod(gps_elapsed_ms, 604_800_000)


    return {
        "lat": round(output.lat * 1e7),
        "lon": round(output.lon * 1e7),
        "yaw": yaw,
        "horiz_accuracy": output.sigma_m,
        "alt": output.alt_m,
        "gps_id": 0,
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



def request_sim_state(connection):
    """Request SIM_STATE telemetry at 2 Hz."""

    connection.mav.command_long_send(
        connection.target_system,
        connection.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0,
        108,       # SIM_STATE message ID
        500_000,   # Interval: 500,000 us = 2 Hz
        0, 0, 0, 0, 0,
    )





def main():
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
    print(f"Map loaded: {map_rgba.shape[1]}x{map_rgba.shape[0]}, GSD={georef.gsd_x:.2f} m/px")

    messages = {}
    last_received = {}

    last_received["HEARTBEAT"] = time.monotonic()

    link_alive = True
    heartbeat_timeout = 3.0

    last_sent = 0.0
    send_interval = 0.2  # 5 Hz

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
            timeout=1,
        )

        

        now = time.monotonic()

        if (
            msg is not None
            and msg.get_type() == "HEARTBEAT"
            and msg.autopilot == mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA
        ):
            if not link_alive:
                print("SITL connection restored!")

                connection.target_system = msg.get_srcSystem()
                connection.target_component = msg.get_srcComponent()

                messages.clear()
                last_received.clear()
                last_sent = 0.0

                # Re-request SIM_STATE after SITL reboot
                request_sim_state(connection)

            last_received["HEARTBEAT"] = now
            link_alive = True

        if now - last_received["HEARTBEAT"] > heartbeat_timeout:
            if link_alive:
                print("WARNING: SITL connection lost!")

                messages.clear()

            link_alive = False

        if not link_alive:
            continue
        
        if msg is not None:
            msg_type = msg.get_type()
            messages[msg_type] = msg

            if msg_type != "HEARTBEAT":
                last_received[msg_type] = time.monotonic()

            if msg_type == "GPS_RAW_INT":
                gps = parse_gps(messages, last_received)
            
                print(
                    f"GPS: available={gps.available}, "
                    f"lat={gps.lat:.7f}, "
                    f"lon={gps.lon:.7f}, "
                    f"alt={gps.alt_m:.2f}"
                )
            

            if msg_type == "SIM_STATE":
                print("TRUE SIM ALT:", msg.alt)


            if msg_type == "SIMSTATE":
                now = time.monotonic()

                if now - last_sent < send_interval:
                    continue

                state = parse_state(messages, last_received)
            
                if state is not None:

                    sim_state = messages.get("SIM_STATE")
                    sim_time = last_received.get("SIM_STATE")

                    if sim_state is None or sim_time is None:
                        continue
                    
                    # Do not use stale simulation data
                    if time.monotonic() - sim_time > 2.0:
                        continue
                    
                    px_x, px_y = latlon_to_px(georef, state.lat, state.lon)
                    heading_deg = math.degrees(state.yaw_rad)
                    scale = compute_scale(state.alt_m, fov_deg, georef.gsd_x, FRAME_SIZE[0])

                    frame = render_frame(map_rgba, px_x, px_y, heading_deg, scale, FRAME_SIZE, augment=True)
                    if frame.shape[2] == 4:
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    else:
                        frame_bgr = frame

                    fix = estimate_pose(frame_bgr, map_bgr, georef)
                    if not fix.ok:
                        print(f"WARNING: matcher failed, skipping GPS_INPUT")
                        continue

                    test_output = ModuleOutput(
                        lat=fix.lat,
                        lon=fix.lon,
                        alt_m=sim_state.alt,
                        heading_rad=fix.heading_rad,
                        sigma_m=fix.sigma_m,
                        module_status=ModuleStatus.FIX,
                    )


                    if test_output.module_status != ModuleStatus.FIX:
                        print("WARNING: No reliable position fix")
                        continue
                    
                    encoded = encode_output(test_output)
                    
                    connection.mav.gps_input_send(**encoded)

                    now = time.monotonic()

                    if last_sent > 0:
                        interval = now - last_sent
                        frequency = 1.0 / interval

                        print(f"GPS_INPUT frequency: {frequency:.2f} Hz")

                    last_sent = now


                    print(
                        "GPS_INPUT prepared:",
                        encoded["lat"],
                        encoded["lon"],
                        "alt:", encoded["alt"],
                        "sigma:", f"{fix.sigma_m:.2f}m"
                    )



        


if __name__ == "__main__":
    main()
