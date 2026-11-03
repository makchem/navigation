# navigation

Experiments in AI-assisted navigation on a map **without GPS** — localizing by
matching what a camera sees against reference map imagery.

## Environment

Managed by [uv](https://docs.astral.sh/uv/). Python 3.12, OpenCV 5.0 (contrib build).

```bash
uv sync                        # create .venv and install everything
uv run scripts/check_env.py    # verify the install end to end
uv run pytest                  # headless regression tests
```

Anything that needs the venv runs under `uv run ...`; there's no need to activate
it manually.

### Verified working on this machine

| Check | Result |
| --- | --- |
| OpenCV | 5.0.0, contrib modules present, 12 threads, SIMD optimizations on |
| Detectors | ORB, SIFT, AKAZE, BRISK all produce descriptors |
| Match + homography | centre reprojection error 0.08 px against a known warp |
| Optical flow | Farneback and Lucas-Kanade both recover a known 4 px shift |
| GUI (`imshow`) | works via the wheel's bundled Qt5 backend |
| Cameras | `/dev/video0` (640×480) and `/dev/video2` (640×360) capture frames |
| Video I/O | FFMPEG enabled |

Live camera sanity check with keypoints drawn on the feed:

```bash
uv run scripts/camera_test.py 0     # q/ESC quits, SPACE saves a frame to out/
```

## Gotchas found during setup

- **OpenCV 5 moved detectors.** `AKAZE`, `KAZE`, and `BRISK` are no longer on the
  top-level `cv2` namespace as they were in 4.x — they now live in
  `cv2.xfeatures2d` (hence the **contrib** wheel). `ORB` and `SIFT` stayed put.
  Tutorials written for 4.x will break on this; `cv2.AKAZE_create()` → 
  `cv2.xfeatures2d.AKAZE.create()`.
- **Never install `opencv-python` and `opencv-contrib-python` together** — they
  overwrite each other's `cv2` module. This project uses only the contrib build.
- **Qt font warning.** The wheel ships Qt5 but no fonts, so `imshow` prints
  `QFontDatabase: Cannot find font directory`. `scripts/fix_qt_fonts.sh` links
  the system DejaVu fonts in to silence it. Re-run it after any `uv sync` that
  reinstalls OpenCV, since `.venv` isn't tracked in git.
- **Wayland notice.** `Ignoring XDG_SESSION_TYPE=wayland on Gnome` is harmless —
  windows render through XWayland. Set `QT_QPA_PLATFORM=wayland` to opt into
  native Wayland if you ever need it.

## Layout

```
src/matching.py     match_frame, estimate_pose
src/simulation.py   fake_drone_frame — synthetic views with known ground truth
src/metrics.py      gsd, error_m
notebooks/          exploration, 01 -> 04
scripts/            env verification + camera utilities
tests/              headless regression tests for the vision stack
data/               map imagery (gitignored)
out/                generated frames and debug output (gitignored)
```

`src/` is a **flat layout installed editable**: `src/matching.py` imports as
top-level `matching` from any working directory, so notebooks need no
`sys.path` juggling. The mapping comes from `sources = ["src"]` in
`pyproject.toml`. Adding a module means dropping a new `.py` in `src/` —
no config change, though a brand-new file needs a kernel restart to be
importable the first time.

Tradeoff worth knowing: top-level names like `metrics` are generic and could
collide with a future dependency. Fine for an exploration project; if it ever
bites, move to `src/dronenav/` and import `from dronenav import metrics`.

## Notebook workflow

```bash
uv run jupyter lab          # kernel is this project's .venv automatically
```

Every notebook opens with the same setup cell: imports, a `DATA` path, a
`show()` helper for inline OpenCV display, and —

```python
%load_ext autoreload
%autoreload 2
```

That last part is the point of this structure. Write a function in
`src/simulation.py`, re-run the notebook cell, and the new code is live with no
kernel restart and no re-loading the 41 MPix map. Notebooks stay as narrative
and plots; anything you want to reuse or test moves into `src/`.

`ipympl` is installed, so `%matplotlib widget` gives you a pan/zoom canvas —
much nicer than re-cropping to inspect a region of a big map.

### The notebooks

| Notebook | Purpose |
| --- | --- |
| `01_explore_map.ipynb` | Map properties, GSD, sample crops, where features actually are |
| `02_template_baseline.ipynb` | `matchTemplate` floor to beat; find where it breaks |
| `03_orb_vs_akaze.ipynb` | Detector comparison on accuracy, robustness, runtime |
| `04_robustness.ipynb` | Failure envelope, negative tests, confidence calibration |

They ship as agendas — markdown prompts with empty code cells — not solutions.

### Reference map

`data/dresden-map.png` is 8192 × 5074 (41.6 MPix), **RGBA**. Two notes:

- `cv2.imread` drops the alpha channel by default. Usually what you want, but
  be deliberate; pass `cv2.IMREAD_UNCHANGED` if you need all four channels.
- Decoded it is only ~0.12 GB, so loading it in a kernel is comfortable.
- If the map spans 5 km across its width, GSD ≈ **0.61 m/px**. Confirm the real
  ground extent — every error figure in metres depends on this one number.

`data/` is gitignored apart from `.gitkeep`, so map imagery stays out of the
repo. Notebook outputs *are* committed as-is; if the diffs get noisy, look at
`nbstripout`.

## What's installed and why

- `opencv-contrib-python` — feature detection, matching, homography, optical flow
- `numpy`, `scipy` — array math, spatial queries (KD-trees for match pruning)
- `matplotlib` — debug plots of tracks and match quality
- `pyproj` — pixel ↔ lat/lon ↔ UTM transforms, for tying matches to real coords
- `jupyterlab`, `ipykernel`, `ipympl` (dev) — the exploration front end

Likely additions once the approach firms up, deliberately left out for now to
keep the install light: `rasterio` (GeoTIFF map tiles), `shapely` (geometry),
`torch` (learned descriptors like SuperPoint/LightGlue, which outperform SIFT for
cross-season and cross-viewpoint map matching).

## ArduPilot SITL

Start

```
cd ~/ardupilot
Tools/autotest/sim_vehicle.py -v ArduPlane --console --map --out=udp:127.0.0.1:14551
```