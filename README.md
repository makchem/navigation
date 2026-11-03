# navmod — GPS-Denied Optical Navigation

Position sensor for ArduPilot that localizes by matching a downward-looking
camera against reference map imagery. Provides cross-check and fallback when
GPS is jammed or spoofed.

## Quick start

```bash
uv sync                        # create .venv and install everything
uv run scripts/check_env.py    # verify OpenCV environment
uv run pytest                  # run tests
uv run jupyter lab              # notebooks
```

## Layout

```
src/navmod/          the navigation module (pip install -e .)
  types.py           State, Fix, GpsMeasurement, ModuleOutput
  georef.py          px <-> metres (ENU) <-> lat/lon
  camera.py          render synthetic nadir frames from the map
  matcher.py         estimate position from frame vs reference map
  metrics.py         GSD, error_m
config/              YAML configs (map georef, camera, scenarios)
scripts/             standalone tools and SITL bridge
notebooks/           exploration and log analysis
tests/               pytest
data/                map imagery (gitignored)
logs/                CSV flight logs (gitignored)
```

Notebooks import `from navmod import matcher, camera, metrics, georef`.
`autoreload 2` is active, so editing `src/navmod/*.py` takes effect on the
next cell run without a kernel restart.

## Environment

Python 3.12, managed by [uv](https://docs.astral.sh/uv/). OpenCV 5.0 (contrib build).

### Verified working

| Check | Result |
| --- | --- |
| OpenCV | 5.0.0, contrib modules present, 12 threads, SIMD on |
| Detectors | ORB, SIFT, AKAZE, BRISK all produce descriptors |
| Match + homography | centre reprojection error 0.08 px against a known warp |
| Optical flow | Farneback and Lucas-Kanade recover a known 4 px shift |

### Gotchas

- **OpenCV 5 moved detectors.** AKAZE/KAZE/BRISK now in `cv2.xfeatures2d`.
- **Never install `opencv-python` alongside `opencv-contrib-python`.**
- **Qt font warning.** Run `scripts/fix_qt_fonts.sh` after any `uv sync`.

## Reference map

`data/dresden-map-partial.png` (1716×1063, RGBA) is the current working map.
The full map (`dresden-map.png`, 8192×5074) needs re-download.

`data/` is gitignored apart from `.gitkeep`.

## ArduPilot SITL

```bash
cd ~/ardupilot
Tools/autotest/sim_vehicle.py -v ArduPlane -N --console --map \
    --out=udp:127.0.0.1:14551 --custom-location=<LAT,LON,ALT,HDG>
```

The SITL bridge (`scripts/bridge.py`) connects the navigation module to
ArduPilot via pymavlink. See CLAUDE.md for integration details.
