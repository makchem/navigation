"""Log SITL truth, secondary GPS (visual matcher) and EKF to CSV.

Usage:
    1. Start sim_vehicle.py with an extra --out=udp:127.0.0.1:14552.
    2. Run: uv run python scripts/flight_compare.py
    3. Inspect logs/flight_compare.csv.

This observes GPS2_RAW as RECEIVED by ArduPilot, not every raw fix
calculated by bridge.py. For original fix values, use bridge.py output.
"""

import csv
import math
import time
from datetime import datetime, timezone
from pathlib import Path

from pymavlink import mavutil

UDP_CONNECTION = "udpin:127.0.0.1:14552"
LOG_PATH = Path("logs/flight_compare.csv")
LOG_INTERVAL_S = 1.0
MAX_AGE_S = 2.0

COLUMNS = [
    "timestamp_utc",
    "gps1_fix_type",
    "true_lat", "true_lon",
    "gps2_fix_type", "gps2_lat", "gps2_lon", "gps2_error_m",
    "ekf_lat", "ekf_lon", "ekf_error_m",
]


def distance_m(lat1, lon1, lat2, lon2):
    """Haversine distance in metres, useful for this local SITL test."""
    earth_radius_m = 6_371_000.0
    a1, a2 = math.radians(lat1), math.radians(lat2)
    dlat = a2 - a1
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(a1) * math.cos(a2) * math.sin(dlon / 2) ** 2)
    return 2 * earth_radius_m * math.asin(min(1.0, math.sqrt(a)))


def fresh(cache, name, now):
    item = cache.get(name)
    if item is None or now - item[1] > MAX_AGE_S:
        return None
    return item[0]


def request_interval(conn, msgid, period_us=200_000):
    conn.mav.command_long_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0, msgid, period_us, 0, 0, 0, 0, 0,
    )


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = mavutil.mavlink_connection(
        UDP_CONNECTION, source_system=246, dialect="ardupilotmega"
    )
    print(f"Listening on {UDP_CONNECTION} ...")
    if conn.wait_heartbeat(timeout=20) is None:
        raise RuntimeError("No SITL heartbeat on port 14552")
    print("Connected; requesting telemetry...")
    for msgid in (164, 24, 124, 33):  # SIMSTATE, GPS_RAW_INT, GPS2_RAW, GLOBAL_POSITION_INT
        request_interval(conn, msgid)

    cache = {}
    last_log = 0.0

    with LOG_PATH.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=COLUMNS)
        writer.writeheader()
        print(f"Writing {LOG_PATH}")
        try:
            while True:
                msg = conn.recv_match(
                    type=["SIMSTATE", "GPS_RAW_INT", "GPS2_RAW", "GLOBAL_POSITION_INT"],
                    blocking=True,
                    timeout=1,
                )
                if msg is None:
                    continue
                now = time.monotonic()
                cache[msg.get_type()] = (msg, now)
                if msg.get_type() != "SIMSTATE" or now - last_log < LOG_INTERVAL_S:
                    continue
                last_log = now

                truth = fresh(cache, "SIMSTATE", now)
                gps1 = fresh(cache, "GPS_RAW_INT", now)
                gps2 = fresh(cache, "GPS2_RAW", now)
                ekf = fresh(cache, "GLOBAL_POSITION_INT", now)
                if truth is None:
                    continue

                true_lat = truth.lat / 1e7
                true_lon = truth.lng / 1e7
                row = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                    "gps1_fix_type": gps1.fix_type if gps1 else "",
                    "true_lat": round(true_lat, 7),
                    "true_lon": round(true_lon, 7),
                    "gps2_fix_type": gps2.fix_type if gps2 else "",
                    "gps2_lat": "", "gps2_lon": "", "gps2_error_m": "",
                    "ekf_lat": "", "ekf_lon": "", "ekf_error_m": "",
                }

                if gps2 and gps2.fix_type >= 3 and (gps2.lat != 0 or gps2.lon != 0):
                    row["gps2_lat"] = gps2.lat / 1e7
                    row["gps2_lon"] = gps2.lon / 1e7
                    row["gps2_error_m"] = round(
                        distance_m(true_lat, true_lon, row["gps2_lat"], row["gps2_lon"]), 2
                    )

                if ekf and (ekf.lat != 0 or ekf.lon != 0):
                    row["ekf_lat"] = ekf.lat / 1e7
                    row["ekf_lon"] = ekf.lon / 1e7
                    row["ekf_error_m"] = round(
                        distance_m(true_lat, true_lon, row["ekf_lat"], row["ekf_lon"]), 2
                    )

                writer.writerow(row)
                outfile.flush()

                def pos(lat, lon):
                    return f"({lat:.7f}, {lon:.7f})" if lat != "" and lon != "" else "no fix"

                print(
                    f"{row['timestamp_utc']} "
                    f"GPS1_fix={row['gps1_fix_type']} "
                    f"TRUE={pos(row['true_lat'], row['true_lon'])} "
                    f"GPS2={pos(row['gps2_lat'], row['gps2_lon'])} "
                    f"err={row['gps2_error_m'] or '-'}m "
                    f"EKF={pos(row['ekf_lat'], row['ekf_lon'])} "
                    f"err={row['ekf_error_m'] or '-'}m"
                )
        except KeyboardInterrupt:
            print(f"\nSaved {LOG_PATH}")


if __name__ == "__main__":
    main()
