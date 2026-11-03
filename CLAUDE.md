# CLAUDE.md — GPS-Denied Optical Navigation Module

## Concept

A position sensor for ArduPilot, not a navigator. It runs during the whole
flight in parallel with GPS:

- A downward-looking camera + IMU continuously estimate position and heading.
- While GPS agrees with the optical estimate, GPS is passed through and optics
  acts as a cross-check.
- If GPS is lost (jamming) or diverges from optics (spoofing, including slow
  drift), the module excludes GPS and feeds position from optics/IMU only.
- GPS is re-enabled only after sustained agreement with optics.
- Output per step: position (lat/lon/alt), heading, uncertainty (σ in metres),
  GPS status (ok/jammed/spoofed), module status (fix/imu_only/unreliable).
- The autopilot decides what to do. The module never makes route decisions.

Key differentiator: **integrity** — calibrated uncertainty. A module that
confidently reports a wrong position is worse than no module.

## Commands

```bash
uv sync                        # install / update deps
uv run pytest                  # run all tests
uv run scripts/check_env.py    # verify OpenCV environment
uv run scripts/check_data.py   # validate map imagery in data/
uv run jupyter lab              # notebooks
```

## Package layout

```
src/navmod/          installable package (pip install -e .)
  __init__.py
  types.py           State, Fix, GpsMeasurement, ModuleOutput
  georef.py          px <-> metres (ENU) <-> lat/lon
  camera.py          render_frame — synthetic nadir views from the map
  matcher.py         estimate_position from frame vs reference map
  metrics.py         GSD, error_m
  odometry.py        (later) frame-to-frame visual odometry
  fusion.py          (later) Kalman filter
  gps_sim.py         (later) GPS scenarios: normal, jammed, spoofed
  monitor.py         (later) GPS vs optics decision, re-enable hysteresis
  mav.py             (later) pymavlink helpers
config/              YAML configs (map georef, camera, scenarios)
scripts/             standalone tools and the SITL bridge
  bridge.py          (later) main SITL loop
  preview.py         (later) live frame viewer
notebooks/           experiments and log analysis
tests/               pytest
data/                map imagery (gitignored)
logs/                CSV flight logs (gitignored)
```

## Conventions

- `navmod` is a pure library: no SITL, no MAVLink, no side effects.
  `scripts/bridge.py` glues SITL to the library.
- Coordinate conversions go through `georef.py` — the single source of truth
  for px ↔ ENU ↔ lat/lon and (h,w) vs (w,h).
- Every estimator returns an uncertainty or explicit failure, never just a
  "best match." The `Fix` dataclass has `ok: bool` and `sigma_m: float`.
- Prefer `cv2.estimateAffinePartial2D` over `findHomography` for nadir
  imagery (4-DOF similarity, not 8-DOF).
- OpenCV 5: AKAZE/KAZE/BRISK are in `cv2.xfeatures2d`, not top-level `cv2`.
- Map images: `cv2.imread` drops alpha by default. The map is RGBA.
- Frame rendering and reference matching must use **separate** source images
  to avoid unrealistically easy matching.

## SITL integration design

`src/navmod/` knows nothing about SITL. `scripts/bridge.py` does:

1. Read ground truth from SIMSTATE (lat/lng in degE7, roll/pitch/yaw in rad).
   Altitude from GLOBAL_POSITION_INT.relative_alt (baro, unaffected by GPS
   spoofing). Do NOT use GLOBAL_POSITION_INT lat/lon as truth.
2. `gps_sim` produces simulated GPS per scenario.
3. `camera` renders the frame from true state.
4. `matcher` computes the optical position + uncertainty.
5. `monitor` compares GPS vs optics, chooses output.
6. Output sent as MAVLink GPS_INPUT (fix_type 3, degE7, horiz_accuracy = σ).
   GPS type parameter must be MAVLink (14). Bridge must be running before arm.
7. Log every step to CSV.

Connection: `pymavlink` (not DroneKit, not MAVSDK).

ArduPilot SITL start command:
```
cd ~/ardupilot
Tools/autotest/sim_vehicle.py -v ArduPlane -N --console --map \
    --out=udp:127.0.0.1:14551 --custom-location=<LAT,LON,ALT,HDG>
```

## Milestones

- M1: Restructure + georef (px↔m↔latlon round-trip tests, types, config) ✓
- M2: Frame rendering from state (render_frame driven by lat/lon/alt/attitude)
- M3: Bridge v0 (fake module: truth + noise via GPS_INPUT, CSV logging)
- M4: Real matcher in the loop (error in metres along a mission)
- M5: GPS scenarios + monitor (jamming, slow spoofing, re-enable)
- M6: Tilt correction, VO, Kalman filter, calibrated uncertainty

## Map

Test map: Google Earth export of Dresden area.
- `data/dresden-map-partial.png`: 1716×1063 RGBA (working subset)
- `data/dresden-map.png`: needs re-download (was truncated)
- Georef corners: **FILL IN** from Google Earth
- If ~5 km wide across 8192 px → GSD ≈ 0.61 m/px
