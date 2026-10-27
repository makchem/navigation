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
src/navigation/     package code
scripts/            env verification + camera utilities
tests/              headless regression tests for the vision stack
data/               map tiles / imagery (gitignored)
out/                generated frames and debug output (gitignored)
```

## What's installed and why

- `opencv-contrib-python` — feature detection, matching, homography, optical flow
- `numpy`, `scipy` — array math, spatial queries (KD-trees for match pruning)
- `matplotlib` — debug plots of tracks and match quality
- `pyproj` — pixel ↔ lat/lon ↔ UTM transforms, for tying matches to real coords

Likely additions once the approach firms up, deliberately left out for now to
keep the install light: `rasterio` (GeoTIFF map tiles), `shapely` (geometry),
`torch` (learned descriptors like SuperPoint/LightGlue, which outperform SIFT for
cross-season and cross-viewpoint map matching).
