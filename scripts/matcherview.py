"""MAVProxy local matcher marker. Display only; never sends MAVLink to ArduPilot.

Put this module on MAVProxy's PYTHONPATH and run `module load matcherview`.
UDP datagrams on loopback 127.0.0.1:15379 come from bridge.py.
"""
import json
import math
import os
import socket
import time

from MAVProxy.modules.lib import mp_module
from MAVProxy.modules.mavproxy_map import mp_slipmap

PORT = int(os.getenv("NAVMOD_MATCHER_MAP_PORT", "15379"))
ICON_KEY = "navmod_raw_matcher_position"


def parse_matcher_packet(raw):
    """Validate untrusted local diagnostics; rejected/invalid points never display."""
    try:
        value = json.loads(raw.decode("utf-8"))
        if value.get("type") != "matcher_fix":
            return None
        lat = float(value["lat"])
        lon = float(value["lon"])
        heading = float(value["heading_deg"])
        frame = int(value["frame_id"])
        if not all(math.isfinite(v) for v in (lat, lon, heading)):
            return None
        if abs(lat) > 90 or abs(lon) > 180 or frame < 0:
            return None
        return (lat, lon, heading % 360.0, frame)
    except (ValueError, KeyError, TypeError, AttributeError, UnicodeError):
        return None


class MatcherViewModule(mp_module.MPModule):
    def __init__(self, mpstate):
        super().__init__(mpstate, "matcherview", "Raw map matcher position (green plane)")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", PORT))
        self.sock.setblocking(False)
        self.last_fix = None
        self.last_received = None
        self.drawn_on_map = None
        self.last_frame = -1
        self.add_command("matcherview", self.cmd_matcherview,
                         "Show last accepted matcher fix", ["status"])
        print("MatcherView listening on 127.0.0.1:%d; "
              "green aircraft is LAST accepted raw matcher fix" % PORT)

    def cmd_matcherview(self, args):
        if args and args[0] != "status":
            print("Usage: matcherview status")
            return
        if self.last_fix is None:
            print("MatcherView: no accepted matcher fix received yet")
        else:
            age = time.monotonic() - self.last_received
            lat, lon, heading, frame = self.last_fix
            print("MatcherView: frame=%d lat=%.7f lon=%.7f "
                  "age=%.1fs heading=%.1f° (LAST FIX, not interpolated)"
                  % (frame, lat, lon, age, heading))

    def idle_task(self):
        # Drain UDP to newest fix; do not block flight telemetry processing.
        latest = None
        for _ in range(128):
            try:
                payload, _addr = self.sock.recvfrom(4096)
            except BlockingIOError:
                break
            except OSError:
                return
            parsed = parse_matcher_packet(payload)
            if parsed is not None:
                latest = parsed
        if latest is not None:
            self.last_fix = latest
            self.last_received = time.monotonic()
        target_map = getattr(self.mpstate, "map", None)
        if target_map is None or self.last_fix is None:
            return
        lat, lon, heading, frame = self.last_fix
        if target_map is not self.drawn_on_map:
            # Use MAVProxy's own green airplane icon, matching the old GPS2 look.
            icon = target_map.icon("greenplane.png")
            target_map.add_object(mp_slipmap.SlipIcon(
                ICON_KEY, (lat, lon), icon, layer=3,
                rotation=heading, follow=False,
                trail=mp_slipmap.SlipTrail(),
            ))
            self.drawn_on_map = target_map
            self.last_frame = frame
        elif latest is not None and frame != self.last_frame:
            target_map.set_position(ICON_KEY, (lat, lon), rotation=heading)
            self.last_frame = frame

    def unload(self):
        self.sock.close()
        target_map = getattr(self.mpstate, "map", None)
        if target_map is not None and self.drawn_on_map is target_map:
            target_map.remove_object(ICON_KEY)
        super().unload()


def init(mpstate):
    return MatcherViewModule(mpstate)
