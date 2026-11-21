# Matcher audit — defects and fixes

## 1. Map features re-extracted every frame

**Problem.** `_match_core` called `detectAndCompute` over the full 8192×5097
Dresden map on every invocation: ~0.72 s extraction + ~0.45 s knnMatch = 1.17 s
per frame, against a 0.20 s budget at 5 Hz.

**Fix.** Added `MapIndex` dataclass holding `points_px` (N,2 float32),
`descriptors`, and `map_shape`, with a `build()` classmethod that extracts ORB
features once. A `query(center_px, radius_px)` method returns only candidates
inside a square window. `estimate_pose` and `match_frame` accept either a
`MapIndex` or a raw `np.ndarray`; for the raw path a one-slot identity cache
(`_map_cache`) reuses the index when the same array object is passed again —
`calibration_harness.py` and `build_sigma_table.py` benefit without code
changes.

## 2. Frame detector sized off the map

**Problem.** `_default_nfeatures(map_img)` was applied to both the map and the
frame. For a 512×512 frame this requested ~28k ORB features, producing ~3900
weak ones. This inflated `n_total` (the denominator of `inlier_ratio`), pushing
the median ratio down to 0.24. The `ok=False` threshold is 0.25, so accurate
frames (actual error 0.28 m) were being discarded.

**Fix.** `FRAME_NFEATURES = 1500` is used for the frame detector. The
area-based sizing stays for the map only. Both images are explicitly converted
to grayscale before feature extraction.

## 3. Searching the whole map instead of a window around the prior

**Problem.** Every frame searched all map features, even when a prior position
was available. At radius 750 px: 24 ms/frame vs 200 ms, p90 error 0.25 m vs
0.63 m, failures 1/40 vs 4/40.

**Fix.** `estimate_pose` and `match_frame` accept `prior_px` and
`search_radius_px` arguments, passed through to `MapIndex.query()`. When both
are provided, only map features inside the square window are matched.

## 4. knnMatch unpacking crash on single-element pairs

**Problem.** `for m, n in matches` raises `ValueError` when a pair has one
element, taking the bridge loop down.

**Fix.** Each pair is length-checked before the Lowe ratio test:

```python
for pair in matches:
    if len(pair) < 2:
        continue
    m, n = pair
    ...
```

## Additional fixes

### `_NO_FIX` singleton → `_no_fix()` factory

The module-level `_NO_FIX` was a mutable `Fix` with `lat=0.0, lon=0.0`. (0,0)
is a valid coordinate ~5800 km from Dresden — a caller that skips the `ok`
check would ship it to the autopilot. And any caller mutating the singleton
would poison it process-wide. Replaced with `_no_fix()` returning `NaN`
lat/lon/heading and `inf` sigma, fresh object each call.

### `_sigma_from_lut` returns 0.0 on reject → inf

Zero sigma reads to EKF3 as a perfect measurement. Now returns `inf`.

### Scale sanity gate with `expected_scale`

The old 0.1–10 range never rejected anything. Added an optional
`expected_scale` argument; a recovered scale outside ±35% is rejected.

---

## Bridge-side integration

`scripts/bridge.py` needs four changes (this module does not touch bridge.py):

### 1. Build the MapIndex once after `load_pipeline`

```python
from navmod.matcher import MapIndex

map_index = MapIndex.build(map_bgr)
```

Pass `map_index` instead of `map_bgr` to `estimate_pose`. This avoids
re-extracting the map on every frame.

### 2. Pass `sigma_lut` from config

The bridge currently never passes `sigma_lut`, so the calibrated table in
`config/default.yaml` is unused and the matcher falls back to hardcoded
interpolation. Load it from config:

```python
sigma_lut = cfg["sigma_lut"]["thresholds"]
fix = estimate_pose(frame, map_index, georef, sigma_lut=sigma_lut, ...)
```

### 3. Pass `prior_px`, `search_radius_px=750`, and `expected_scale`

```python
fix = estimate_pose(
    frame, map_index, georef,
    sigma_lut=sigma_lut,
    prior_px=prior_px,
    search_radius_px=750,
    expected_scale=render_scale,
)
```

**Critical: `prior_px` must come from the module's own last accepted fix plus
dead reckoning, never from `state.lat`/`state.lon`.** `state` is SIMSTATE
ground truth — seeding the prior from truth would make the GPS-denied demo
measure nothing (the matcher would always search the right window and always
converge, but it would never demonstrate GPS-independent navigation).

Recommended prior management:

- **Seed** from the last valid GPS position at startup.
- **Update** from each accepted fix (`fix.ok=True`), converting lat/lon back to
  map pixel via `latlon_to_px`.
- **Widen the radius** after consecutive failures (e.g. 750 → 1500 → 3000).
- **Fall back to global search** (`prior_px=None`) after a few seconds without
  a fix.

### 4. Pass `expected_scale`

`expected_scale` is the render scale implied by the altitude prior: the same
value passed to `camera.render_frame(scale=...)`. The matcher rejects a
recovered scale outside ±35% of this value.

---

## Follow-ups

1. **Sigma LUT recalibration.** The LUT in `config/default.yaml` was calibrated
   with global search. Windowed search shifts the inlier ratio distribution
   substantially (median 0.24 → 0.51 on harsh augmentation). Re-run
   `scripts/build_sigma_table.py` once the bridge uses windowed search.

2. **`test_augmented_heading` calls `pytest.skip` instead of asserting.**
   `tests/test_matcher.py::test_augmented_heading` skips when the matcher fails
   to converge — the test covering exactly this failure mode keeps the suite
   green and should assert instead.

3. **Unused import in `georef.py`.** `georef.py` imports
   `pyproj.Transformer` and never uses it.
